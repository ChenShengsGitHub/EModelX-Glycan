# 环境
import torch
import os
import argparse
import torch
import mrcfile
import gemmi
import superpose3d
import numpy as np


from torch import nn
from glob import glob
from tqdm import tqdm
from copy import deepcopy
from tqdm import trange ,tqdm
from src.unet.unet3d import UNet3d
from src.util_misc import calc_dis
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from src.util_em import normalize_map
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from subprocess import run, DEVNULL
from scipy.optimize import linear_sum_assignment

def get_res_conform(res,mrc_tensor,temp_dict,c1_idx=None):
    coords_tar=[]
    coords_ori=[]
    ring_atom_ids=['C1','C2','C3','C4','C5','O5']
    for atom_id in ring_atom_ids:
        
        coords_ori.append(temp_dict['XYP'][atom_id].get_coord())
    if c1_idx is  None:
        for atom_id in ring_atom_ids:
            coords_tar.append(res[atom_id].get_coord())
    else:
        tar_atom_ids=ring_atom_ids[c1_idx::1]+ring_atom_ids[:c1_idx:1]
        for atom_id in tar_atom_ids:
            coords_tar.append(res[atom_id].get_coord())

    rmsd,R,T,_=superpose3d.Superpose3D(coords_tar,coords_ori)
    best_score=-10000
    best_res=None
    for res_name in temp_dict:
        # print(res_name)
        this_res=temp_dict[res_name]
        coords=[]
        for atom in this_res:
            coords.append(np.dot(atom.get_coord(),R.T)+T)
        score=get_pixel_value(coords,mrc_tensor).mean().cpu().numpy()
        if score>best_score:
            best_score=score
            best_res=res_name
    print(best_res)

    new_residue = Residue(res.id, best_res[:3], '')
    for atom in temp_dict[best_res]:
        atom_id=atom.id
        if atom_id=='O1':
            continue
        atom_coord=np.dot(atom.get_coord(),R.T)+T
        atom = Atom(
            name=atom_id, 
            coord=atom_coord, 
            bfactor=0.0, 
            occupancy=1.0, 
            altloc=' ', 
            fullname=" " + atom_id.ljust(3), 
            serial_number=len(new_residue)+1, 
            element=atom_id[0]
        )
        new_residue.add(atom)
    return new_residue

def get_pixel_value(coords,mrc_tensor):
    coords_tensor=torch.tensor(np.array(coords)).view(1,1,1,-1,3).cuda()

    # 计算 coords_tensor 的像素值，仿照 @file_context_0 的实现
    coords_flat = coords_tensor.squeeze(1).reshape(-1, 3).cpu().numpy()  # [N, 3]
    _,_,D, H, W = mrc_tensor.shape

    grid = np.zeros_like(coords_flat)
    coords_flat_np = np.asarray(coords_flat)
    grid[:, 0] = 2.0 * coords_flat_np[:, 2] / (W-1) - 1.0  # x
    grid[:, 1] = 2.0 * coords_flat_np[:, 1] / (H-1) - 1.0  # y
    grid[:, 2] = 2.0 * coords_flat_np[:, 0] / (D-1) - 1.0  # z

    grid_tensor = torch.from_numpy(grid).float().to('cuda')
    grid_tensor = grid_tensor.unsqueeze(1).unsqueeze(1)             # [N, 1, 1, 3]
    grid_tensor = grid_tensor.permute(1, 0, 2, 3)                   # [1, N, 1, 3]
    grid_tensor = grid_tensor.unsqueeze(2)                          # [1, N, 1, 1, 3]

    sampled_vals2 = torch.nn.functional.grid_sample(
        mrc_tensor.to('cuda'), grid_tensor.to('cuda'), mode='bilinear', align_corners=True
    )
    return sampled_vals2

def coord_to_index(coord):
    return tuple(np.round(coord).astype(int))

def is_connected_by_density(mrc_data, res1, res2, min_len=6, max_len=20,max_dis=6,ign_aid1=None):
    """
    判断两个残基在密度中是否有连通（使用广度优先搜索，step步长），只能走过密度高于阈值的格点。
    """
    import numpy as np
    from collections import deque

    # coords1 = np.array([atom.get_coord() for atom in res1])
    # coords2 = np.array([atom.get_coord() for atom in res2])
    ring_atom_ids=['C1','C2','C3','C4','C5','O5']
    ring_atom_mrc_values=[]
    for atom_id in ring_atom_ids:
        try:
            ring_atom_mrc_values.append(mrc_data[coord_to_index(res1[atom_id].get_coord())])
        except:
            pass
        try:
            ring_atom_mrc_values.append(mrc_data[coord_to_index(res2[atom_id].get_coord())])
        except:
            pass
    # print(ring_atom_mrc_values)
    ring_atom_mrc_values=np.array(ring_atom_mrc_values)
    threshold=ring_atom_mrc_values.mean()/3
    # print(threshold)
    

    shape = mrc_data.shape

    max_search_len = max_len
    min_search_len = min_len

    directions = []
    for dx in range(-1,2):
        for dy in range(-1,2):
            for dz in range(-1,2):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                if np.linalg.norm((dx,dy,dz))>1.1:
                    continue
                directions.append((dx,dy,dz))
    
    coords1=[]
    coords2=[]
    for aid in ring_atom_ids:
        coords1.append(res1[aid].get_coord())
        coords2.append(res2[aid].get_coord())
    cross_dis=calc_dis(coords1,coords2)
    if ign_aid1 is not None:
        cross_dis[ign_aid1]=10000
    min_dis=np.min(cross_dis)
    if min_dis>max_dis:
        return []
    candidates=np.where(cross_dis<(min_dis+1.4))

    # 遍历res1和res2的所有原子对
    valid_routes=[]
    for i,j in zip(candidates[0],candidates[1]):
        aid1=ring_atom_ids[i]
        aid2=ring_atom_ids[j]
        atom1=res1[aid1]
        atom2=res2[aid2]
        distance=np.linalg.norm(atom1.get_coord()-atom2.get_coord())
        if distance>max_dis:
            continue
        start = tuple(np.round(atom1.get_coord()).astype(int))
        end = tuple(np.round(atom2.get_coord()).astype(int))
        # mid should be the coordinate starting from atom2 towards atom1 for a length of 1.43
        ac2=atom1.get_coord()
        ac=atom2.get_coord()
        direction = ac2 - ac
        direction_norm = np.linalg.norm(direction)
        if direction_norm >=2.86:
            unit_direction = direction / direction_norm
            mid = ac + unit_direction * 1.43
        else:
            mid = (ac+ac2)/2

        visited = np.zeros(shape, dtype=bool)
        queue = deque()
        queue.append( (start, [], 0,start) )
        visited[start] = True

        while queue:
            current, score_list,moves,midle = queue.popleft()
            if moves > max_search_len:
                continue
            # 找到终点了
            if np.linalg.norm(np.array(current) - np.array(end)) <= 0.5:
                # print(aid1,aid2,steps,distance)
                if moves <= min_search_len and moves-distance<1:
                    # print(score_list)
                    valid_routes.append((aid1,aid2,moves,distance,mid))
                break
            for d in directions:
                next_pos = tuple(np.array(current)+np.array(d))
                # 边界检查
                if any([next_pos[i]<0 or next_pos[i]>=shape[i] for i in range(3)]):
                    continue
                if visited[next_pos]:
                    continue
                # 密度阈值判断
                if not np.linalg.norm(np.array(next_pos) - np.array(end)) <= 0.5:
                    if not moves==0:
                        if mrc_data[next_pos] < threshold:
                            continue
                visited[next_pos] = True
                score_list.append(mrc_data[next_pos])
                if np.linalg.norm(np.array(next_pos)-np.array(atom1.get_coord())) <= np.linalg.norm(np.array(next_pos) - np.array(end)):
                    new_moves=np.linalg.norm(np.array(next_pos)-np.array(atom1.get_coord()))
                    midle=next_pos
                    # print(np.linalg.norm(np.array(next_pos)-np.array(atom1.get_coord())))
                elif np.linalg.norm(np.array(next_pos) - np.array(end)) <= 0.5:
                    new_moves=moves+np.linalg.norm(np.array(midle)-np.array(atom2.get_coord()))
                    # print(np.linalg.norm(np.array(current)-np.array(atom2.get_coord())))
                else:
                    new_moves=moves
                queue.append((next_pos, deepcopy(score_list),float(new_moves),midle))
    return valid_routes

import argparse
import ast

def parse_args():
    parser = argparse.ArgumentParser(description='Auto build configuration for NP5')
    parser.add_argument('--task', type=str, required=True, help='Task name')
    parser.add_argument('--map_fn', type=str, required=True, help='Map file name')
    parser.add_argument('--origin_map', type=str, default='', help='Original map file')
    parser.add_argument('--o_link_pdb', type=str, default='', help='Linked PDB file')
    parser.add_argument('--o_link_site', type=str, default="", help="O link site as tuple string")
    parser.add_argument('--cand_num', type=int, default=2000, help='Number of candidates')
    parser.add_argument('--resol', type=float, required=True, help='Resolution value')
    parser.add_argument('--cp1', type=str, default='data/model/find_ch-epoch=08-val_recall=1.00-val_precision=0.408.ckpt', help='CP1 file name')
    parser.add_argument('--cp2', type=str, default='data/model/ring_cls-epoch=00-val_recall=0.98-val_precision=0.038.ckpt', help='CP2 file name')
    return parser.parse_args()

def parse_o_link_site(site_str):
    # safely evaluate tuple string
    try:
        return ast.literal_eval(site_str)
    except Exception as e:
        return []

args = parse_args()




# config={
# }


# # Update config['NP5'] with CLI arguments
# config['NP5'] = {
#     'origin_map': args.origin_map,
#     'map_fn': args.map_fn,
#     'o_link_pdb': args.o_link_pdb,
#     'o_link_site': parse_o_link_site(args.o_link_site),
#     'refined_fn': args.refined_fn,
#     'cleaned_fn': args.cleaned_fn,
#     'cand_num': args.cand_num,
#     'resol': args.resol
# }

# 配置
task=args.task
cp1=args.cp1
cp2=args.cp2
cand_num=args.cand_num
resol=args.resol
if task == 'build_bond':
    map_fn=args.map_fn
    origin_map=args.origin_map
    if not origin_map:
        raise ValueError('origin_map is required for build_bond task')
    o_link_pdb=args.o_link_pdb
    if not o_link_pdb:
        raise ValueError('o_link_pdb is required for build_bond task')
    o_link_site=parse_o_link_site(args.o_link_site)
    if not o_link_site:
        raise ValueError('o_link_site is required for build_bond task')
elif task == 'no_bond':
    map_fn=args.map_fn
    o_link_site=parse_o_link_site(args.o_link_site)
else:
    raise ValueError('Invalid task name')


phenix_sh='phenix.sh'
phenix_act_fn='phenix-1.20.1-4487/phenix_env.sh'
phenix_param='data/phenix_refine.eff'

# load so3 samples
# 基于
temp_coords_dict={}
i=0
for file in tqdm(glob('data/so3_sample/*.cif')):
    i+=1
    # print(file)
    # if i<7:
    #     continue
    
    if file.split('/')[-1].split('.')[0] in ['XYP']:
        mmcifParser=MMCIFParser(QUIET=True)
        nag_temp=mmcifParser.get_structure('',file)
        nag_temp=nag_temp[0]
        temp_coords=[]
        temp_coords=[]
        for chain in nag_temp:
            for res in chain:
                temp_resname=res.get_resname()
                tc=[]
                tc_all=[]
                temp_atom_ids=[]
                temp_atom_ids=[]
                for atom in res:
                    atom_id=atom.id
                    if not atom_id.startswith('H'):
                        tc.append(atom.get_coord())
                        temp_atom_ids.append(atom_id)
                ring_coords=[]
                if temp_resname in ['GZL','FUB','AHR']:
                    for atom_id in ['O1','C1','C2','C3','C4']:
                        ring_coords.append(res[atom_id].get_coord())
                else:
                    for atom_id in ['O1','C1','C2','C3','C4','C5']:
                        ring_coords.append(res[atom_id].get_coord())

                tc.append(np.mean(ring_coords,axis=0))
                temp_atom_ids.append('ring_center')
                temp_coords.append(tc)
        # print(np.array(temp_coords).shape)
        temp_coords=torch.tensor(np.array(temp_coords)).view(1,1,-1,len(temp_atom_ids),3).cuda()
        temp_coords_dict[temp_resname]=(temp_atom_ids,temp_coords)

weight_dict={}
ring_idx_dict={}
for file in tqdm(glob('data/monomer/*.cif')):
    fn=file.split('/')[-1]
    cifParser=MMCIFParser(QUIET=True)
    st=cifParser.get_structure('',file)
    weight_list=[]
    ring_idx=[]
    for model in st:
        for chain in model:
            for res in chain:
                res_name=res.get_resname()
                for atom in res:
                    if atom.element!='H':
                        if res_name in ['AHR','FUB','GZL']:
                            # if atom.id in ['O1','C1','C2','C3','C4','O4']:
                            if atom.id in ['O1','C1','C2','C3','C4']:
                                ring_idx.append(len(weight_list))
                                weight_list.append(1/6)
                            elif atom.id in ['O2','O3','O4']:
                                weight_list.append(0)
                        else:
                            # if atom.id in ['O1','C1','C2','C3','C4','C5','O5']:
                            if atom.id in ['O1','C1','C2','C3','C4','C5']:
                                ring_idx.append(len(weight_list))
                                weight_list.append(1/7)
                            elif atom.id in ['O2','O3','O4','O5']:
                                weight_list.append(0)
    weight_list=np.array(weight_list)
    weight_list[weight_list==0]=0.5/(len(weight_list)-np.sum(weight_list!=0))
    weight_dict[res_name]=torch.tensor(weight_list)
    ring_idx_dict[res_name]=np.array(ring_idx)

temp_dict={}
for file in tqdm(glob('data/mono_sugars/*.cif')):
    fn=file.split('/')[-1]
    cifParser=MMCIFParser(QUIET=True)
    st=cifParser.get_structure('',file)
    res_name=file.split('/')[-1].split('.')[0]
    for model in st:
        for chain in model:
            for res in chain:
                temp_dict[res_name]=res

# nn pred
# TODO p99的参照物有问题
model1=UNet3d.load_from_checkpoint(
            checkpoint_path=cp2,
            map_location = {'cuda:7': 'cuda:0'},
            lr=1e-4,
            precision='16',
            label_smoothing=0.1,
            num_levels=4,
            f_maps=64,
            loss_reduction='mean',
            task='ring_cls',
            loss_weight=True
            ).emmodel.cuda().half()
            
model1.eval()

model2=UNet3d.load_from_checkpoint(
        checkpoint_path=cp1,
        map_location = {'cuda:7': 'cuda:0'},
        lr=1e-4,
        precision='16',
        label_smoothing=0.1,
        num_levels=4,
        f_maps=64,
        loss_reduction='mean',
        task='find_ch',
        loss_weight=True
        ).emmodel.cuda().half()
        
model2.eval()

def pred_ring_cls(sub_mrc):
    sub_mrc = np.expand_dims(np.expand_dims(sub_mrc, axis=0), axis=0)
    sub_mrc = torch.from_numpy(sub_mrc).cuda()
    with torch.no_grad():
        output = model1(sub_mrc)
        output2 = model2(sub_mrc)
        output=nn.Softmax(dim=1)(output)
        output=torch.cat([output[:,:2],output[:,2:3]+output[:,3:4]],dim=1)
        score= -output[0,0]+0.3*output[0,1]+output[0,2]
        score[score<0]=0
        output2=nn.Softmax(dim=1)(output2[:,1:])
        result=score*output2[0,1]
    return result.cpu().numpy(), (output[0,1]+output[0,2]).cpu().numpy()

voxel_size=1.5
mrc_fn='/'.join(map_fn.split('/')[:-1]+[f'norm{voxel_size}_'+map_fn.split('/')[-1]])
mrc_obj=normalize_map(map_fn,cif_fn=None,mrc_fn=mrc_fn,target_voxel_size=voxel_size)
EMmap = mrcfile.open(mrc_fn)
mrc_data=np.transpose(EMmap.data.copy(), [2,1,0])

EMmap2=mrcfile.open(mrc_fn)
mrc_data2=np.transpose(EMmap2.data.copy(), [2,1,0])
median_value=max(np.median(mrc_data2),0)
mrc_data2[mrc_data2<median_value]=0
p99=np.percentile(mrc_data2[mrc_data2>0],99)

median_value=max(np.median(mrc_data),0)
mrc_data[mrc_data<median_value]=0
p99=np.percentile(mrc_data[mrc_data>0],99)

if mrc_data.shape[0]<64 or mrc_data.shape[1]<64 or mrc_data.shape[2]<64:
    mrc_data = np.pad(mrc_data, [(0, max(0,64-mrc_data.shape[0])), (0, max(0,64-mrc_data.shape[1])), (0, max(0,64-mrc_data.shape[2]))], 'constant', constant_values=[(0, 0), (0, 0), (0, 0)])
mrc_data=mrc_data/p99
mrc_data=mrc_data.astype(np.float16)
mrc_mean=np.mean(mrc_data[mrc_data>mrc_data.mean()])

vist_data=np.zeros(mrc_data.shape,dtype=np.int16)
mrc_data_copy=mrc_data.copy()
pred_atom=np.zeros(mrc_data.shape,dtype=np.float16)
pred_inring=np.zeros(mrc_data.shape,dtype=np.float16)

while mrc_data_copy.max()>mrc_mean:
    print(mrc_data.shape,mrc_mean,mrc_data_copy.max())
    max_ind=np.argmax(mrc_data_copy)
    max_ind=np.unravel_index(max_ind,mrc_data_copy.shape)
    x,y,z=max_ind
    if x-32<0:
        min_x,max_x=0,64
    elif x+32>mrc_data.shape[0]:
        min_x,max_x=mrc_data.shape[0]-64,mrc_data.shape[0]
    else:
        min_x,max_x=x-32,x+32

    if y-32<0:
        min_y,max_y=0,64
    elif y+32>mrc_data.shape[1]:
        min_y,max_y=mrc_data.shape[1]-64,mrc_data.shape[1]
    else:
        min_y,max_y=y-32,y+32

    if z-32<0:
        min_z,max_z=0,64
    elif z+32>mrc_data.shape[2]:
        min_z,max_z=mrc_data.shape[2]-64,mrc_data.shape[2]
    else:
        min_z,max_z=z-32,z+32
    result,ATProb=pred_ring_cls(mrc_data[min_x:max_x,min_y:max_y,min_z:max_z])
    pred_atom[min_x:max_x,min_y:max_y,min_z:max_z]= \
        (pred_atom[min_x:max_x,min_y:max_y,min_z:max_z]
        *vist_data[min_x:max_x,min_y:max_y,min_z:max_z]
        +ATProb
        )/(vist_data[min_x:max_x,min_y:max_y,min_z:max_z]+1)
    
    pred_inring[min_x:max_x,min_y:max_y,min_z:max_z]= \
        (pred_inring[min_x:max_x,min_y:max_y,min_z:max_z]
        *vist_data[min_x:max_x,min_y:max_y,min_z:max_z]
        +result
        )/(vist_data[min_x:max_x,min_y:max_y,min_z:max_z]+1)
    

    vist_data[min_x:max_x,min_y:max_y,min_z:max_z]+=1
    mrc_data_copy[min_x:max_x,min_y:max_y,min_z:max_z]=0
    # break

pred_inring=pred_inring.astype(np.float32)
pred_atom=pred_atom.astype(np.float32)
mrc_mask=deepcopy(mrc_data)
mrc_mask/=mrc_mask[pred_inring>0.3].mean()
mrc_mask[mrc_mask>1]=1
pred_inring=pred_inring*mrc_mask



CHProb=pred_inring.astype(np.float32)
CH_score_thrh=0.01
pcd_numpy = np.array(np.where(CHProb > CH_score_thrh)).T
while pcd_numpy.shape[0] > cand_num:
    print('CH_score_thrh:',CH_score_thrh,'pcds:',pcd_numpy.shape[0])
    CH_score_thrh += 0.001
    pcd_numpy = np.array(np.where(CHProb > CH_score_thrh)).T
print('CH_score_thrh:',CH_score_thrh,'pcds:',pcd_numpy.shape[0])

# 为pcd_numpy按CHProb[pcd_numpy]从大到小排序
scores = CHProb[pcd_numpy[:,0], pcd_numpy[:,1], pcd_numpy[:,2]]
sort_idx = np.argsort(-scores)
pcd_numpy = pcd_numpy[sort_idx]*voxel_size

voxel_size=1
mrc_fn='/'.join(map_fn.split('/')[:-1]+[f'norm{voxel_size}_'+map_fn.split('/')[-1]])
mrc_obj=normalize_map(map_fn,cif_fn=None,mrc_fn=mrc_fn,target_voxel_size=voxel_size)
EMmap = mrcfile.open(mrc_fn)
mrc_data=np.transpose(EMmap.data.copy(), [2,1,0])

EMmap2=mrcfile.open(mrc_fn)
mrc_data2=np.transpose(EMmap2.data.copy(), [2,1,0])
median_value=max(np.median(mrc_data2),0)
mrc_data2[mrc_data2<median_value]=0
p99=np.percentile(mrc_data2[mrc_data2>0],99)


# median_value=max(np.median(mrc_data),0)
mrc_data[mrc_data<median_value]=0
# p99=np.percentile(mrc_data[mrc_data>0],99)


if mrc_data.shape[0]<64 or mrc_data.shape[1]<64 or mrc_data.shape[2]<64:
    mrc_data = np.pad(mrc_data, [(0, max(0,64-mrc_data.shape[0])), (0, max(0,64-mrc_data.shape[1])), (0, max(0,64-mrc_data.shape[2]))], 'constant', constant_values=[(0, 0), (0, 0), (0, 0)])
mrc_data=mrc_data/p99
mrc_data=mrc_data.astype(np.float16)
mrc_mean=np.mean(mrc_data[mrc_data>mrc_data.mean()])

offset=np.array([EMmap.header["origin"].x,EMmap.header["origin"].y,EMmap.header["origin"].z])
# CH_cand_map=eval(pred_inring.astype(np.float32),mrc_data.astype(np.float32),deepcopy(EMmap.header["origin"]),deepcopy(EMmap.voxel_size),pred_furanose,pred_pyranose,prefix,args.cif,{'A':{'THP':['O06','O08']}},args)
print(offset)

mask = mrc_data > mrc_mean
mrc_data[mask] = mrc_mean + (mrc_data[mask] - mrc_mean) * 0.1


mrc_tensor=torch.from_numpy(mrc_data).unsqueeze(0).unsqueeze(0).float()
# atom_tensor=torch.from_numpy(pred_atom).unsqueeze(0).unsqueeze(0).float()

from src.util_misc import calc_dis
self_dis=calc_dis(pcd_numpy,pcd_numpy)
# 对于每个index，获取与其self_dis<8的index list
neighbor_indices_list = []
threshold = 10
for idx in range(self_dis.shape[0]):
    neighbors = np.where(self_dis[idx] < threshold)[0]
    neighbor_indices_list.append(neighbors.tolist())


# density fitting

batch_size = 10
# voxel_size = 1.5
num_points = pcd_numpy.shape[0]
max_value_results = []
arg_ind_results=[]
resname_results=[]
for i in trange(0, num_points, batch_size):
    batch_pcd = pcd_numpy[i:min(i+batch_size, num_points)]  # shape: [B, 3]
    B = batch_pcd.shape[0]
    batch_pcd_float = batch_pcd.astype(np.float32)

    max_value_dict={}
    max_ind_dict={}
    for res_name in temp_coords_dict:
        temp_atom_ids,temp_coords=temp_coords_dict[res_name]
        temp_coords_norm = temp_coords.cpu() / voxel_size  # [1, 1, A, C, 3]
        # batch_pcd_float: [B, 3]
        batch_pcd_float_tensor = torch.from_numpy(batch_pcd_float).to(torch.float32)  # [B, 3]
        # Expand batch_pcd_float_tensor to [B, 1, 1, 1, 3] for broadcasting
        batch_pcd_float_tensor = batch_pcd_float_tensor[:, None, None, None, :]  # [B, 1, 1, 1, 3]
        # temp_coords_norm: [1, 1, A, C, 3]，此时已是tensor不用转
        if not isinstance(temp_coords_norm, torch.Tensor):
            temp_coords_norm = torch.from_numpy(temp_coords_norm).to(torch.float32)
        coords = batch_pcd_float_tensor + temp_coords_norm  # [B, 1, A, C, 3]
        coords = coords.squeeze(1)  # [B, A, C, 3]
        # coords: [B, A, C, 3]
        coords_flat = coords.reshape(-1, 3)  # [B*A*C, 3]
        D, H, W = mrc_data.shape
        grid = np.zeros_like(coords_flat)
        # Avoid DeprecationWarning by ensuring coords_flat is a proper numpy array (not a subclass)
        coords_flat_np = np.asarray(coords_flat)
        grid[:, 0] = 2.0 * coords_flat_np[:, 2] / (W-1) - 1.0  # x
        grid[:, 1] = 2.0 * coords_flat_np[:, 1] / (H-1) - 1.0  # y
        grid[:, 2] = 2.0 * coords_flat_np[:, 0] / (D-1) - 1.0  # z
        grid_tensor = torch.from_numpy(grid).float().to('cuda')
        # grid_sample 需要 [1, N, 1, 1, 3]，N=B*A*C
        grid_tensor = grid_tensor.unsqueeze(1).unsqueeze(1)  # [N, 1, 1, 3]
        grid_tensor = grid_tensor.permute(1, 0, 2, 3)        # [1, N, 1, 3]
        grid_tensor = grid_tensor.unsqueeze(2)               # [1, N, 1, 1, 3]
        sampled_vals = torch.nn.functional.grid_sample(
            mrc_tensor.to('cuda'), grid_tensor.to('cuda'), mode='bilinear', align_corners=True
        )  # [1, 1, N, 1, 1]
        sampled_vals = sampled_vals.squeeze(0).squeeze(0).squeeze(-1).squeeze(-1)  # [N]
        # temp_coords: [1, 1, A, C, 3]，所以每个点有A*C个原子
        A = temp_coords.shape[2]
        C = temp_coords.shape[3]
        sampled_vals = sampled_vals.view(B, A, C)  # [B, A, C]
        sampled_vals[:,:,-1]=sampled_vals[:,:,-1]*-1
        
        weighted_sum = sampled_vals.mean(dim=-1) - sampled_vals[:,:,ring_idx_dict[res_name]].std(dim=-1)  # [B, A]

        max_value, arg_ind = torch.max(weighted_sum, dim=-1)
        max_value_dict[res_name]=max_value
        max_ind_dict[res_name]=arg_ind
    
    res_names = list(max_value_dict.keys())
    stacked_max_values = torch.stack([max_value_dict[res_name] for res_name in res_names], dim=0)  # [num_res, B]
    stacked_arg_inds = torch.stack([max_ind_dict[res_name] for res_name in res_names], dim=0)      # [num_res, B]
    # 取每个B上最大的max_value及其res_name和arg_ind
    max_vals, res_idx = torch.max(stacked_max_values, dim=0)  # [B]
    arg_inds = stacked_arg_inds[res_idx, torch.arange(stacked_arg_inds.shape[1])]  # [B]
    # resname_results = [res_names[i] for i in res_idx.cpu().numpy()]
    max_value_results.append(max_vals.cpu().numpy())
    arg_ind_results.append(arg_inds.cpu().numpy())
    resname_results.append([res_names[i] for i in res_idx.cpu().numpy()])

final_max_value_results = np.concatenate(max_value_results, axis=0)  # [num_points]
final_arg_ind_results = np.concatenate(arg_ind_results, axis=0)  # [num_points]
final_resname_results = np.concatenate(resname_results, axis=0)  # [num_points]
print("final_result shape:", final_max_value_results.shape)

sorted_indices = np.argsort(-final_max_value_results)

mean_1=final_max_value_results.mean()
mean_mrc=final_max_value_results[final_max_value_results>mean_1].mean()
print(mean_1,mean_mrc)

# initial model building
built_coords=[]
built_pcd_i=[]
built_scores=[]
built_pvs=[]
chain=Chain('A')
for i in trange(final_max_value_results.shape[0]):
    pcd_i=sorted_indices[i]
    this_coord=pcd_numpy[pcd_i,:]
    max_value=final_max_value_results[pcd_i]
    if max_value< mean_mrc*0.8:
        continue
    res_name=final_resname_results[pcd_i]
    temp_atom_ids,temp_coords=temp_coords_dict[res_name]

    best_coord=temp_coords[0,0,final_arg_ind_results[pcd_i],:,:].cpu().numpy()
    new_residue = Residue((' ', len(chain)+1, ' '), res_name, '')
    coords=[]
    all_coords=[]
    for k in range(len(temp_atom_ids)):
        atom_id=temp_atom_ids[k]
        if len(atom_id)>3:
            continue
        atom_coord=best_coord[k,:]
        my_coord=atom_coord+offset+this_coord*voxel_size
        atom = Atom(
            name=atom_id, 
            coord=my_coord, 
            bfactor=1/(final_max_value_results[pcd_i]+1e-6), 
            occupancy=1.0, 
            altloc=' ', 
            fullname=" " + atom_id.ljust(3), 
            serial_number=len(new_residue)+1, 
            element=atom_id[0]
        )
        new_residue.add(atom)
        if atom_id.startswith('H'):
            continue
        if res_name in ['AHR','FUB','GZL']:
            if atom_id in ['C1','C2','C3','C4','O4']:
                coords.append(my_coord)
        else:
            if atom_id in ['C1','C2','C3','C4','C5','O5']:
                coords.append(my_coord)
        all_coords.append(my_coord)
    # 计算neighbor_indices_list，即与pcd_numpy[pcd_i]的坐标距离小于8的所有pcd_numpy的index
    threshold = 6
    # 只计算pcd_i这一行的距离，更快
    self_dis = calc_dis([pcd_numpy[pcd_i]], pcd_numpy)[0]  # shape: [num_points]
    neighbor_indices = np.where(self_dis < threshold)[0]
    # 用于后续循环一致性，包装成list
    neighbor_indices_list = neighbor_indices.tolist()
    flag=True
    pv=get_pixel_value(coords,mrc_tensor).mean().cpu().numpy()
    for neighbor_idx in neighbor_indices_list:
        # print(neighbor_idx)
        if neighbor_idx in built_pcd_i:
            id_this=built_pcd_i.index(neighbor_idx)
            the_built_coords=built_coords[id_this]
            the_built_scores=built_scores[id_this]
            the_built_pvs=built_pvs[id_this]
            cross_dis=calc_dis(coords,the_built_coords)
            min_cross_dis_all=calc_dis(all_coords,the_built_coords).min(axis=-1)
            if np.sum(cross_dis<2)>2 or np.sum(cross_dis<1)>0:
                flag=False
                break
            elif np.sum(min_cross_dis_all<1.5)>1:
                flag=False
                break
            elif max_value < the_built_scores*0.5 or pv<the_built_pvs*0.5:
                flag=False
                break
                
            
    if flag:
        built_coords.append(coords)
        built_pcd_i.append(pcd_i)
        built_scores.append(max_value)
        built_pvs.append(pv)
        chain.add(new_residue)

model=Model(0)
struct=Structure('0')
struct.add(model)
model.add(chain)
mmcif_io=MMCIFIO()
mmcif_io.set_structure(struct)
fn=map_fn.split('/')[-1].split('.')[0]
map_dir = '/'.join(map_fn.split('/')[:-1])
initial_fn=os.path.join(map_dir,f'final_{fn}_initial.cif')
mmcif_io.save(initial_fn)

cmd_in=f"phenix.real_space_refine {os.path.abspath(initial_fn)} {os.path.abspath(map_fn)} {os.path.abspath(phenix_param)}"
for key in temp_coords_dict:
    cmd_in+=f" {os.path.abspath(f'data/elbow_mono_sugar/{key}.cif')}"
cmd_in+=f" resolution={resol}"
cmd=f'bash {phenix_sh} {phenix_act_fn} {map_dir} "{cmd_in}"'
print(cmd)
run([cmd],  shell=True)

if task == 'build_bond':
    # TODO run phenix (with automatic covalent linking == False) and PROVIDE YOUR refined_fn HERE!
    refined_fn=os.path.join(map_dir,f'final_{fn}_initial_real_space_refined_000.cif')
    try:
        refined_st=MMCIFParser(QUIET=True).get_structure('refined',refined_fn)
    except:
        refined_st=PDBParser(QUIET=True).get_structure('refined',refined_fn)


    coords=[]
    for atom in refined_st.get_atoms():
        coords.append(atom.get_coord())
    mean_mrc=get_pixel_value(coords,mrc_tensor).mean().cpu().numpy()
    print(mean_mrc)

    # remove unfit sugar monomers in refined_st
    new_chain=Chain('A')
    for res in refined_st.get_residues():
        coords=[]
        for atom in res.get_atoms():
            if atom.id in ['C1','C2','C3','C4','C5','O5']:
                coords.append(atom.get_coord())
        sampled_vals2=get_pixel_value(coords,mrc_tensor)
        mean_value=sampled_vals2.mean()
        print(res.get_id(),torch.sum(sampled_vals2>mean_mrc*0.5))
        if torch.sum(sampled_vals2>mean_mrc*0.5)<4:
            continue
        new_chain.add(res)

    model=Model(0)
    struct=Structure('0')
    struct.add(model)
    model.add(new_chain)
    mmcif_io=MMCIFIO()
    mmcif_io.set_structure(struct)
    fn=map_fn.split('/')[-1].split('.')[0]
    map_dir = '/'.join(map_fn.split('/')[:-1])
    new_fn=os.path.join(map_dir,f'final_{fn}_cleaned.cif')
    mmcif_io.save(new_fn)

    cmd_in=f"phenix.real_space_refine {os.path.abspath(new_fn)} {os.path.abspath(origin_map)} {os.path.abspath(phenix_param)}"
    for key in temp_coords_dict:
        cmd_in+=f" {os.path.abspath(f'data/elbow_mono_sugar/{key}.cif')}"
    cmd_in+=f" resolution={resol}"
    cmd=f'bash {phenix_sh} {phenix_act_fn} {map_dir} "{cmd_in}"'
    print(cmd)
    run([cmd],  shell=True)

    # TODO run phenix again (with previously saved model) and PROVIDE YOUR refined_fn HERE!
    # cleaned_fn='data/inhouse/HYP1_0916_refined_1_real_space_refined_058.cif'
    # cleaned_fn='data/inhouse/HYP2_0916_refined_1_real_space_refined_061.cif'
    # cleaned_fn='data/inhouse/SER_0916_refined_1_real_space_refined_055.cif'

    # temp_dict={'XYP':temp_dict['XYP'],'XYP_rot':temp_dict['XYP_rot'],'BGC':temp_dict['BGC'],'BGC_rot':temp_dict['BGC_rot']}
    cleaned_fn=os.path.join(map_dir,f'final_{fn}_cleaned_real_space_refined_000.cif')
    try:    
        refined_st=MMCIFParser(QUIET=True).get_structure('refined',cleaned_fn)
    except:
        refined_st=PDBParser(QUIET=True).get_structure('refined',cleaned_fn)

    try:    
        o_link_st=MMCIFParser(QUIET=True).get_structure('o_link',o_link_pdb)
    except:
        o_link_st=PDBParser(QUIET=True).get_structure('o_link',o_link_pdb)

    o_link_coords=[]
    cid,rid=o_link_site
    # for res in o_link_st[0][cid]:
    #     print(res.get_id())
    for atom in o_link_st[0][cid][rid]:
        o_link_coords.append(atom.get_coord())

    o_link_mean_coord=np.mean(o_link_coords,axis=0)
    print(o_link_mean_coord)

    start_coords=[o_link_mean_coord]
    start_rids=[None]
    start_aids=[None]

    new_chain=Chain('A')
    # for res in refined_st.get_residues():
    #     print(res.get_id())
    unselected_residues=[deepcopy(res) for res in refined_st[0]['A']]
    print(len(unselected_residues))
    # 列表按索引删除，可以用del语句，例如:
    # del unselected_residues[index]


    current_residues=[]
    bond_info=[]
    ring_atom_ids=['C1','C2','C3','C4','C5','O5']
    connect_atom_ids=['C2','C3','C4','C5','C6']
    xyp_coords=[]
    for aid in ring_atom_ids:
        xyp_coords.append(temp_dict['XYP'][aid].get_coord())
    xyp_coords=np.array(xyp_coords)

    try:
        del temp_dict['AHR']
        del temp_dict['AHR_rot']
        del temp_dict['FUB']
        del temp_dict['FUB_rot']
        del temp_dict['GZL']
        del temp_dict['GZL_rot']
    except:
        pass

    conn_aids_dict={}
    for temp_key in temp_dict:
        conn_aids=set([])
        for aid in ['C2','C3','C4','C5','C6']:
            oid='O'+aid[1:]
            flag=True
            if not (aid and oid in temp_dict[temp_key]):
                flag=False
            if temp_dict[temp_key].get_resname() in ['AHR','FUB','GZL']:
                if aid=='C4':
                    flag=False
            else:
                if aid=='C5':
                    flag=False
            if flag:
                conn_aids.add(oid)
        conn_aids_dict[temp_key]=list(conn_aids)

    while unselected_residues:
        if not current_residues:
            print('restart...')
            best_res=None
            best_score=-100000
            best_res_idx=None
            best_c1_idx=None
            best_pre_rid=None
            best_pre_aid=None
            for i,res in enumerate(unselected_residues):
                coords=[]
                for atom_id in ring_atom_ids:
                    coords.append(res[atom_id].get_coord())
                coords=np.array(coords)

                dismap=calc_dis(start_coords,coords)
                abs_dis_143=np.min(np.abs(dismap-1.43))

                # 取dismap上最小的10个pair，并获取他们的行列index
                abs_dis_143_flat = abs_dis_143.flatten()
                candidates=np.where(dismap<(6-1.43))
                if candidates[0].shape[0]==0:
                    min_indices_sorted = np.argsort(abs_dis_143_flat)[:6]
                    top6_row_idx, top6_col_idx = np.unravel_index(min_indices_sorted, dismap.shape)
                    candidates=(top6_row_idx,top6_col_idx)
                score_list=[]
                for ii,jj in zip(candidates[0],candidates[1]):
                    p1 = start_coords[ii]
                    p2 = coords[jj]
                    quint_points = [p1 + (p2 - p1) * t for t in np.linspace(0, 1, 11)[2:-2]]
                    score=get_pixel_value(quint_points,mrc_tensor).mean().cpu().numpy()
                    error=0 if dismap[ii,jj]<3 else dismap[ii,jj]-3
                    score=score-error
                    score_list.append(score)
                max_score_idx = np.argmax(score_list)
                max_score=score_list[max_score_idx]

                if max_score>best_score:
                    best_score=max_score
                    best_res=res
                    best_res_idx=i
                    best_c1_idx=candidates[1][max_score_idx]
                    best_pre_rid=start_rids[candidates[0][max_score_idx]]
                    best_pre_aid=start_aids[candidates[0][max_score_idx]]
            if best_pre_rid is not None:
                print(best_pre_rid,best_res.get_id())
            current_residues.append((best_res,best_c1_idx))
            unselected_residues.pop(best_res_idx)
            if not unselected_residues:
                new_res=get_res_conform(best_res,mrc_tensor,temp_dict,best_c1_idx)
                # print(new_res.get_resname())
                new_chain.add(new_res)
                break

        # print(len(unselected_residues),best_res_idx)
        for res,c1_idx in current_residues:
            res_id=res.get_id()
            print(res_id)

        new_current_residues=[]
        new_current_residues_score=[]
        for res,old_c1_idx in current_residues:
            c1_score_list=[]
            max_key_list=[]
            c1_idx_list=[]
            results_list=[]
            R_list=[]
            T_list=[]
            for c1_offset in range(-1,2):
                c1_idx=(old_c1_idx+c1_offset)%6
                # print('c1_idx',c1_idx,c1_offset)
                res_id=res.get_id()
                
                target_atom_ids=ring_atom_ids[c1_idx:]+ring_atom_ids[:c1_idx]
                target_coords=[]
                for atom_id in target_atom_ids:
                    target_coords.append(res[atom_id].get_coord())
                target_coords=np.array(target_coords)
                rmsd,R,T,_=superpose3d.Superpose3D(target_coords,xyp_coords)

                match_dict={}
                for ix,res2 in enumerate(unselected_residues):
                    if res2.get_id()==res.get_id():
                        continue
                    connected_atoms=is_connected_by_density(mrc_data,res,res2,min_len=5,ign_aid1=c1_idx)
                    if connected_atoms:
                        for item in connected_atoms:
                            if item[0] not in match_dict:
                                match_dict[item[0]]=[]
                            match_dict[item[0]].append((res2.get_id(),item[1],item[2],item[3],ix,item[4]))
                        # print(res_id,res2.get_id())
                        # print(res2.get_id(),connected_atoms)
                # print(match_dict)
                matches=[[]]
                used_resids=[[]]
                for aid in match_dict:
                    new_matches=[]
                    new_used_resids=[]
                    for match,uid in zip(matches,used_resids):
                        for item2 in match_dict[aid]:
                            if item2[0] in uid:
                                continue
                            new_matches.append(match+[item2])
                            new_used_resids.append(uid+[item2[0]])
                        new_matches.append(match+[None])
                        new_used_resids.append(uid+[None])
                    matches=new_matches
                    used_resids=new_used_resids
                score_list=[]
                for match in matches:
                    score=0
                    for m in match:
                        if m is not None:
                            score+=1000
                            eroor=m[2]**2+m[3]**2
                            score-=eroor
                    # print(match)
                    score_list.append(score)
                max_score_idx = np.argmax(score_list)
                max_score=score_list[max_score_idx]
                # print(max_score)
                tar_coords=[]
                tar_rid_aid=[]
                for k,m in zip(match_dict.keys(),matches[max_score_idx]):
                    if m is not None:
                        print(k,m)
                        coord=refined_st[0]['A'][m[0]][m[1]].get_coord()
                        tar_coords.append(coord)
                        tar_rid_aid.append((m[4],m[1]))
                

                # b=1/0
                
                results = {key: [] for key in temp_dict}
                temp_score = {key: [] for key in temp_dict}
                for temp_key in temp_dict:
                    temp_resname=temp_key[:3]
                    coords_transed=[]
                    for aid in connect_atom_ids:
                        oid='O'+aid[1:]
                        flag=True
                        if not (aid and oid in temp_dict[temp_key]):
                            flag=False
                        if temp_resname in ['AHR','FUB','GZL']:
                            if aid=='C4':
                                flag=False
                        else:
                            if aid=='C5':
                                flag=False
                        if flag:
                            coords_transed.append(np.dot(temp_dict[temp_key][aid].get_coord(),R.T)+T)
                        else:
                            coords_transed.append(np.array([-10000,-10000,-10000]))

                    coords_transed_all=[]
                    for atom in temp_dict[temp_key]:
                        coords_transed_all.append(np.dot(np.array(atom.get_coord()),R.T)+T)
                    coords_transed_all=np.array(coords_transed_all)
                    temp_score[temp_key]=[get_pixel_value(coords_transed_all,mrc_tensor).mean().cpu().numpy()/mean_mrc]
                    for bond in bond_info:
                        if bond[2]== res_id:
                            prev_coord=new_chain[bond[0]][bond[1]].get_coord()
                            dist=np.linalg.norm(prev_coord-res[ring_atom_ids[c1_idx]].get_coord())
                            temp_score[temp_key].append(-dist**2)

                    if tar_coords:
                        conn_aids=conn_aids_dict[temp_key]
                        src_coords=[]
                        for aid in conn_aids:
                            src_coords.append(np.dot(temp_dict[temp_key][aid].get_coord(),R.T)+T)
                        src_coords=np.array(src_coords)

                        dismap=calc_dis(tar_coords,src_coords)
                        # INSERT_YOUR_CODE
                        # 为 dismap 计算二分图最大匹配
                        

                        # 这里假设我们要求最小距离的匹配（匈牙利算法求最小权匹配）
                        row_ind, col_ind = linear_sum_assignment(dismap)
                        

                        # 匹配的距离
                        matched_distances = dismap[row_ind, col_ind][:dismap.shape[0]]
                        
                        
                        # 保存匹配结果到 results[temp_key]，可以保存为 (tar_idx, src_idx, distance)
                        for t_idx, s_idx, dist in zip(row_ind, col_ind, matched_distances):
                            results[temp_key].append((conn_aids_dict[temp_key][s_idx], tar_rid_aid[t_idx][0], ring_atom_ids.index(tar_rid_aid[t_idx][1]), 1/(dist+1)))
                            temp_score[temp_key].append(-dist**2)

                    


                score_dict={}
                for key2 in temp_score:
                    score_dict[key2]=np.sum(temp_score[key2])
                max_key_this = max(score_dict, key=score_dict.get)
                c1_score=len(tar_coords)*10000+score_dict[max_key_this]
                print(c1_score)
                c1_score_list.append(c1_score)
                max_key_list.append(max_key_this)
                c1_idx_list.append(c1_idx)
                results_list.append(results)
                R_list.append(R)
                T_list.append(T)
                # print(max_key,score_dict[max_key])
                # n=1/0
            # INSERT_YOUR_CODE
            if np.max(c1_score_list)<100:
                c1_offset_idx=1
            else:
                c1_offset_idx=np.argmax(c1_score_list)
            max_key=max_key_list[c1_offset_idx]
            c1_idx=c1_idx_list[c1_offset_idx]
            results=results_list[c1_offset_idx]
            R=R_list[c1_offset_idx]
            T=T_list[c1_offset_idx]


            conn_aids=set([])
            for aid in ['C2','C3','C4','C5','C6']:
                oid='O'+aid[1:]
                flag=True
                if not (aid and oid in temp_dict[max_key]):
                    flag=False
                if temp_dict[max_key].get_resname() in ['AHR','FUB','GZL']:
                    if aid=='C4':
                        flag=False
                else:
                    if aid=='C5':
                        flag=False
                if flag:
                    conn_aids.add(aid)

            temp_res=deepcopy(temp_dict[max_key])
            new_residue = Residue(res.id, max_key[:3], '')
            for atom in temp_dict[max_key]:
                atom_id=atom.id
                if atom_id=='O1':
                    continue
                atom_coord=np.dot(atom.get_coord(),R.T)+T
                atom = Atom(
                    name=atom_id, 
                    coord=atom_coord, 
                    bfactor=0.0, 
                    occupancy=1.0, 
                    altloc=' ', 
                    fullname=" " + atom_id.ljust(3), 
                    serial_number=len(new_residue)+1, 
                    element=atom_id[0]
                )
                new_residue.add(atom)
            
            for result in results[max_key]:
                # if res_id==(' ', 15, ' '):
                #     print(result,unselected_residues[result[1]].get_id())
                new_current_residues.append((unselected_residues[result[1]],result[2]))
                print('children',unselected_residues[result[1]].get_id())
                new_current_residues_score.append(result[3])
                bond_info.append((res_id,result[0],unselected_residues[result[1]].get_id(),'C1'))
                atom_id='O'+result[0][1:]
                car_id='C'+result[0][1:]
                ac=new_residue[car_id].get_coord()
                ac2=unselected_residues[result[1]][ring_atom_ids[result[2]]].get_coord()
                # ac3=ac向ac2方向移动1.43长度
                direction = ac2 - ac
                direction_norm = np.linalg.norm(direction)
                if direction_norm >=2.86:
                    unit_direction = direction / direction_norm
                    ac3 = ac + unit_direction * 1.43
                else:
                    ac3 = (ac+ac2)/2

                new_residue[atom_id].set_coord(ac3)
                conn_aids.discard(result[0])
                # print(result,(res_id,ring_atom_ids[result[0]],unselected_residues[result[1]].get_id(),'C1'))
            
            # 为了避免pop导致顺序变化，先记录要删除的索引，然后从大到小删除
            indices_to_remove = sorted([result[1] for result in results[max_key]], reverse=True)
            for idx in indices_to_remove:
                # print('remove',unselected_residues[idx].get_id())
                unselected_residues.pop(idx)
            

            new_chain.add(new_residue)
            for aid in conn_aids:
                start_coords.append(new_residue[aid].get_coord())
                start_rids.append(new_residue.get_id())
                start_aids.append(aid)

        if new_current_residues_score:
            # 获取排序后的索引
            this_sorted_indices = sorted(range(len(new_current_residues_score)), key=lambda i: new_current_residues_score[i], reverse=True)
            # 依排序重排new_current_residues和new_current_residues_score
            current_residues = [new_current_residues[i] for i in this_sorted_indices]
            
        else:
            current_residues = []
        
        if not unselected_residues:
            for res,c1_idx in current_residues:
                new_res=get_res_conform(res,mrc_tensor,temp_dict,c1_idx)
                # print(new_res.get_resname())
                new_chain.add(new_res)



                
    model=Model(0)
    struct=Structure('0')
    struct.add(model)
    model.add(new_chain)
    mmcif_io=MMCIFIO()
    mmcif_io.set_structure(struct)
    fn=map_fn.split('/')[-1].split('.')[0]
    map_dir = '/'.join(map_fn.split('/')[:-1])
    connected_fn=os.path.join(map_dir,f'final_{fn}_conn_with_type.cif')
    mmcif_io.save(connected_fn)



    doc = gemmi.cif.read_file(connected_fn)

    block = doc.sole_block()
    label_seq_ids = block.find_loop('_atom_site.label_seq_id')
    atom_auth_seq_ids = block.find_loop('_atom_site.auth_seq_id')
    atom_auth_to_label = {}
    for label_seq_id,atom_auth_seq_id in zip(label_seq_ids,atom_auth_seq_ids):
        atom_auth_to_label[int(atom_auth_seq_id)] = int(label_seq_id)
    # atom_label_seq_id = block.find('_atom_site.label_seq_id')
    # atom_auth_seq_id = block.find('_atom_site.auth_seq_id')

    # if atom_label_seq_id and atom_auth_seq_id:
    #     for lbl, auth in zip(atom_label_seq_id, atom_auth_seq_id):
    #         # 跳过缺失的映射或者占位
    #         if lbl != '?' and auth != '?':
    #             atom_auth_to_label[auth] = lbl
    # print(comp_id)


    # 建立ptnr1_auth_seq_id和ptnr1_label_seq_id的一一映射关系
    # 提取ATOM行的auth_seq_id和label_seq_id，建立映射

    print(atom_auth_to_label)


    # 写 struct_conn loop
    loop = block.init_loop(
                "_struct_conn.",
                ["id","conn_type_id",
                "ptnr1_label_asym_id","ptnr1_label_comp_id","ptnr1_label_seq_id","ptnr1_label_atom_id",
                "ptnr2_label_asym_id","ptnr2_label_comp_id","ptnr2_label_seq_id","ptnr2_label_atom_id",
                "pdbx_value_order","details"]
            )
    for k,bond in enumerate(bond_info):
        print(bond)
        rid1=bond[0][1]
        rid2=bond[2][1]
        resname1=new_chain[rid1].get_resname()
        resname2=new_chain[rid2].get_resname()
        loop.add_row([f"{k+1}","covalent",'A',resname2,f'{atom_auth_to_label[bond[2][1]]}',bond[3],'A',resname1,f'{atom_auth_to_label[bond[0][1]]}','O'+bond[1][1:],"SING","cryomg"])

    doc.write_file(connected_fn)


    param_fn=os.path.join(map_dir,f'final_{fn}_conn_with_type.param')
    # print(bond_info)
    with open(param_fn,'w') as f:
        # INSERT_YOUR_CODE
        with open(phenix_param, 'r') as eff_file:
            f.write(eff_file.read())
            f.write('\n')
        for k,bond in enumerate(bond_info):
            rid1=bond[0][1]
            rid2=bond[2][1]
            resname1=new_chain[rid1].get_resname()
            resname2=new_chain[rid2].get_resname()
            oid2='O4' if resname2 in ['AHR','FUB','GZL'] else 'O5'
            restraint='\n'.join([
                        # '# ori_bond_length:{:.2f}'.format(p[4]),
                        'refinement.geometry_restraints.edits {',
                        f'  {resname1}{bond[0][1]}_selection = chain A and resname {resname1} and resid {bond[0][1]} and name O{bond[1][1:]}',
                        f'  {resname2}{bond[2][1]}_selection1 = chain A and resname {resname2} and resid {bond[2][1]} and name C1',
                        f'  {resname2}{bond[2][1]}_selection2 = chain A and resname {resname2} and resid {bond[2][1]} and name {oid2}',
                        f'  {resname2}{bond[2][1]}_selection3 = chain A and resname {resname2} and resid {bond[2][1]} and name C2',
                        '  bond {',
                        '    action = *add',
                        f'    atom_selection_1 = ${resname1}{bond[0][1]}_selection',
                        f'    atom_selection_2 = ${resname2}{bond[2][1]}_selection1',
                        '    symmetry_operation = None',
                        '    distance_ideal = 1.43',
                        '    sigma = 0.05',
                        '    slack = None',
                        '  }',
                        '  angle {',
                        '    action = *add',
                        f'    atom_selection_1 = ${resname1}{bond[0][1]}_selection',
                        f'    atom_selection_2 = ${resname2}{bond[2][1]}_selection1',
                        f'    atom_selection_3 = ${resname2}{bond[2][1]}_selection2',
                        '    angle_ideal = 109.9',
                        '    sigma = 3',
                        '  }',
                        '  angle {',
                        '    action = *add',
                        f'    atom_selection_1 = ${resname1}{bond[0][1]}_selection',
                        f'    atom_selection_2 = ${resname2}{bond[2][1]}_selection1',
                        f'    atom_selection_3 = ${resname2}{bond[2][1]}_selection3',
                        '    angle_ideal = 109.9',
                        '    sigma = 3',
                        '  }',
                        '}',
                    ])
                    # print(restraint)
            f.write(restraint+'\n')

                
    cmd_in=f"phenix.real_space_refine {os.path.abspath(connected_fn)} {os.path.abspath(origin_map)} {os.path.abspath(param_fn)}"
    for key in weight_dict:
        cmd_in+=f" {os.path.abspath(f'data/elbow_mono_sugar/{key}.cif')}"
    cmd_in+=f" resolution={resol}"
    cmd=f'bash {phenix_sh} {phenix_act_fn} {map_dir} "{cmd_in}"'
    print(cmd)
    run([cmd],  shell=True)

