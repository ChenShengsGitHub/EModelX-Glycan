import os
import math
import random
import mrcfile
import pickle
import torch

import numpy as np
from copy import deepcopy

from tqdm import tqdm
from collections import namedtuple
from Bio.PDB.MMCIFParser import MMCIFParser

from torch.nn.functional import grid_sample

from .util_em import MRCObject, save_mrc


from .constants import rings_cls, atom_cls, ligand_rings, atom_types


random.seed(2025)

Result = namedtuple('Result', ['mrc_fn','cif_fn','offset','voxel_size','mrc_mean','mrc_max','p999','p99','atom_density','data'])
SubData = namedtuple('SubData', ['raw_cls','label_atom'])

def save_dataset(inp):
    pdbid,mrc_fn,cif_fn,training_dir=inp
    print(pdbid)
    try:
        # with open(ring_json_fn, 'r') as f:
        #     ring_coords = json.load(f)

        EMmap = mrcfile.open(mrc_fn)
        offset = np.array([float(EMmap.header["origin"].x), float(EMmap.header["origin"].y), float(EMmap.header["origin"].z)])
        voxel_size = np.array([EMmap.voxel_size.z,EMmap.voxel_size.y,EMmap.voxel_size.x])
        mrc_data=np.transpose(EMmap.data.copy(), [2,1,0])
        if np.sum(np.isnan(mrc_data))>0:
            print(f"Error: mrc_data has nan in {mrc_fn}")
            return

        median_value=max(np.median(mrc_data),0)
        mrc_data[mrc_data<median_value]=0
        # mrc_data[mrc_data>median_value]-=median_value
        mrc_max=np.max(mrc_data)
        if mrc_max==0:
            print(f"Error: mrc_data is all 0 in {mrc_fn}")
            return
        # mrc_max=np.max(mrc_data)
        # p9999=np.percentile(mrc_data[mrc_data>0],99.99)
        p999=np.percentile(mrc_data[mrc_data>0],99.9)
        p99=np.percentile(mrc_data[mrc_data>0],99)
        if mrc_data.shape[0]<64 or mrc_data.shape[1]<64 or mrc_data.shape[2]<64:
            mrc_data = np.pad(mrc_data, [(0, max(0,64-mrc_data.shape[0])), (0, max(0,64-mrc_data.shape[1])), (0, max(0,64-mrc_data.shape[2]))], 'constant', constant_values=[(0, 0), (0, 0), (0, 0)])
        s0,s1,s2=mrc_data.shape
        mrc_mean=np.mean(mrc_data)
        
        if voxel_size[0]==0 or voxel_size[1]==0 or voxel_size[2]==0:
            print(f"Error: voxel_size is 0 in {mrc_fn}")
            return

        parser = MMCIFParser(QUIET=True)
        model = parser.get_structure('', cif_fn)[0]
    except Exception as e:
        print(f"Error: {e}")
        return
    
    cls_data = np.zeros(mrc_data.shape,dtype=np.int8) # rings_cls['non_atom']
    label_atom = np.zeros(mrc_data.shape,dtype=np.int8) # rings_cls['non_atom']
    dens_list=[]
    for res in model.get_residues():
        resname=res.get_resname()
        ring_atoms=ligand_rings[resname] if resname in ligand_rings else {}
        
        fur_coords=[]
        pyr_coords=[]
        if ring_atoms:
            if len(ring_atoms['in_furanose'])==5:
                for atom_id in ring_atoms['in_furanose']:
                    if atom_id in res:
                        coord = res[atom_id].get_coord()
                        fur_coords.append(coord)

            if len(ring_atoms['in_pyranose'])==6:
                for atom_id in ring_atoms['in_pyranose']:
                    if atom_id in res:
                        coord = res[atom_id].get_coord()
                        pyr_coords.append(coord)

            fur_center,pyr_center=None,None
            if len(fur_coords)==5:
                fur_center=np.mean(fur_coords,axis=0)

            if len(pyr_coords)==6:
                pyr_center=np.mean(pyr_coords,axis=0)

            for mc in [fur_center,pyr_center]:
                if mc is not None:
                    x,y,z = np.round((mc - offset)/voxel_size).astype(int)
                    if x<0 or x>=s0 or y<0 or y>=s1 or z<0 or z>=s2:
                        continue
                    cls_data[x,y,z] = rings_cls['ch_mass_center'] # the max value

        for atom in res.get_atoms():
            coord = atom.get_coord()
            x,y,z = np.round((coord - offset)/voxel_size).astype(int)
            if x<0 or x>=s0 or y<0 or y>=s1 or z<0 or z>=s2:
                continue
            dens_list.append(mrc_data[x,y,z])
            label_atom[x,y,z] = max(label_atom[x,y,z],atom_cls['atom_out_ring_4_7'])

            atom_id=atom.get_id()
            if not ring_atoms:
                cls_data[x,y,z] = max(cls_data[x,y,z],rings_cls['atom_out_ring_4_7'])
            elif atom_id in ring_atoms['in_pyranose']:
                cls_data[x,y,z] = max(cls_data[x,y,z],rings_cls['in_pyranose'])
            elif atom_id in ring_atoms['in_norm_na']:
                cls_data[x,y,z] = max(cls_data[x,y,z],rings_cls['in_norm_na'])
            elif atom_id in ring_atoms['in_furanose']:
                cls_data[x,y,z] = max(cls_data[x,y,z],rings_cls['in_furanose'])
            elif atom_id in ring_atoms['in_fused_ring']:
                cls_data[x,y,z] = max(cls_data[x,y,z],rings_cls['in_fused_ring'])
            elif atom_id in ring_atoms['in_ring_4_7']:
                cls_data[x,y,z] = max(cls_data[x,y,z],rings_cls['in_ring_4_7'])
            else:
                cls_data[x,y,z] = max(cls_data[x,y,z],rings_cls['atom_out_ring_4_7'])

    atom_density=np.mean(dens_list)
    
    for x,y,z in np.argwhere(cls_data>=rings_cls['in_ring_4_7']):
        min_x=x-1 if x>0 else 0
        max_x=x+1 if x<s0-1 else s0-1
        min_y=y-1 if y>0 else 0
        max_y=y+1 if y<s1-1 else s1-1
        min_z=z-1 if z>0 else 0
        max_z=z+1 if z<s2-1 else s2-1
        label_atom[min_x:max_x+1,min_y:max_y+1,min_z:max_z+1] = atom_cls['in_ring_4_7_box333']
    
    label_atom[cls_data>=rings_cls['in_ring_4_7']] = atom_cls['in_ring_4_7']
    
    # label_atom=np.transpose(label_atom, [2,1,0])
    # mrc_obj=MRCObject(grid=label_atom,voxel_size=deepcopy(EMmap.voxel_size),global_origin=deepcopy(EMmap.header["origin"]),mapc=1,mapr=2,maps=3)
    # save_mrc(mrc_obj,f'temp/mrc/{pdbid}_label.mrc')
    
    fold_x=math.ceil(s0/64)
    fold_y=math.ceil(s1/64)
    fold_z=math.ceil(s2/64)
    start_x_list=[0] if fold_x==1 else [round((s0-64)*_/(fold_x-1)) for _ in range(fold_x)]
    start_y_list=[0] if fold_y==1 else [round((s1-64)*_/(fold_y-1)) for _ in range(fold_y)]
    start_z_list=[0] if fold_z==1 else [round((s2-64)*_/(fold_z-1)) for _ in range(fold_z)]

    
    sub_data_dict={}
    for start_x in start_x_list:
        for start_y in start_y_list:
            for start_z in start_z_list:
                sub_mean=np.mean(mrc_data[start_x:start_x+64,start_y:start_y+64,start_z:start_z+64])
                if sub_mean/mrc_mean > random.random():
                    sub_cls = cls_data[start_x:start_x+64,start_y:start_y+64,start_z:start_z+64]
                    sub_label_atom = label_atom[start_x:start_x+64,start_y:start_y+64,start_z:start_z+64]
                    sub_data_dict[(start_x,start_y,start_z)]=SubData(sub_cls,sub_label_atom)
    # print(len(start_x_list)*len(start_y_list)*len(start_z_list),len(sub_data_dict))

    result = Result(mrc_fn,cif_fn,offset,voxel_size,mrc_mean,mrc_max,p999,p99,atom_density,sub_data_dict)
    with open(os.path.join(training_dir,f'{pdbid}.pkl'), 'wb') as f:
        pickle.dump(result, f)
    return 'success'
    
    # mrc_data_copy=mrc_data.copy()
    # sub_data_dict={}
    # count=0
    # split_count=np.ceil(s0/64)*np.ceil(s1/64)*np.ceil(s2/64)
    # while np.max(mrc_data_copy)>mrc_mean and len(sub_data_dict)<np.ceil(split_count/2):
    #     x,y,z=np.unravel_index(np.argmax(mrc_data_copy),mrc_data_copy.shape)
    #     dx=random.randint(16,48)
    #     dy=random.randint(16,48)
    #     dz=random.randint(16,48)
        
    #     if x-dx<0:
    #         min_x,max_x=0,64
    #     elif x+(64-dx)>s0:
    #         min_x,max_x=s0-64,s0
    #     else:
    #         min_x,max_x=x-dx,x+(64-dx)

    #     if y-dy<0:
    #         min_y,max_y=0,64
    #     elif y+(64-dy)>s1:
    #         min_y,max_y=s1-64,s1
    #     else:
    #         min_y,max_y=y-dy,y+(64-dy)

    #     if z-dz<0:  
    #         min_z,max_z=0,64
    #     elif z+(64-dz)>s2:
    #         min_z,max_z=s2-64,s2
    #     else:
    #         min_z,max_z=z-dz,z+(64-dz)
    #     sub_mean=np.mean(mrc_data_copy[min_x:max_x,min_y:max_y,min_z:max_z])
    #     if sub_mean/mrc_mean > random.random():
    #         sub_cls = cls_data[min_x:max_x,min_y:max_y,min_z:max_z]
    #         sub_label_atom = label_atom[min_x:max_x,min_y:max_y,min_z:max_z]
    #         sub_data_dict[(min_x,min_y,min_z)]=SubData(sub_cls,sub_label_atom)
            
    #     mrc_data_copy[min_x:max_x,min_y:max_y,min_z:max_z]=0
    #     count+=1
ResultDiff = namedtuple('ResultDiff', ['mrc_fn','cif_fn','offset','voxel_size','mrc_mean','mrc_max','p999','p99','origin_grid_size','data'])
SubMap = namedtuple('SubMap', ['sub_map','res_name','atom_coord','atom_type'])

def save_coords(inp):
    pdbid,mrc_fn,cif_fn,training_dir,half=inp
    print(pdbid)
    try:
        # with open(ring_json_fn, 'r') as f:
        #     ring_coords = json.load(f)

        EMmap = mrcfile.open(mrc_fn)
        offset = np.array([float(EMmap.header["origin"].x), float(EMmap.header["origin"].y), float(EMmap.header["origin"].z)])
        voxel_size = np.array([EMmap.voxel_size.z,EMmap.voxel_size.y,EMmap.voxel_size.x])
        mrc_data=np.transpose(EMmap.data.copy(), [2,1,0])
        if np.sum(np.isnan(mrc_data))>0:
            print(f"Error: mrc_data has nan in {mrc_fn}")
            return

        median_value=max(np.median(mrc_data),0)
        mrc_data[mrc_data<median_value]=0
        mrc_max=np.max(mrc_data)
        if mrc_max==0:
            print(f"Error: mrc_data is all 0 in {mrc_fn}")
            return
        # mrc_max=np.max(mrc_data)
        # p9999=np.percentile(mrc_data[mrc_data>0],99.99)
        p999=np.percentile(mrc_data[mrc_data>0],99.9)
        p99=np.percentile(mrc_data[mrc_data>0],99)
        if mrc_data.shape[0]<64 or mrc_data.shape[1]<64 or mrc_data.shape[2]<64:
            mrc_data = np.pad(mrc_data, [(0, max(0,64-mrc_data.shape[0])), (0, max(0,64-mrc_data.shape[1])), (0, max(0,64-mrc_data.shape[2]))], 'constant', constant_values=[(0, 0), (0, 0), (0, 0)])
        s0,s1,s2=mrc_data.shape
        mrc_mean=np.mean(mrc_data)
        
        if voxel_size[0]==0 or voxel_size[1]==0 or voxel_size[2]==0:
            print(f"Error: voxel_size is 0 in {mrc_fn}")
            return

        parser = MMCIFParser(QUIET=True)
        model = parser.get_structure('', cif_fn)[0]
    except Exception as e:
        print(f"Error: {e}")
        return
    
    data=[]
    sub_map_list=[]
    val_list=[]
    for res in model.get_residues():
        resname=res.get_resname()
        ring_atoms=ligand_rings[resname] if resname in ligand_rings else {}


        ch_coords=[]
        
        if ring_atoms:
            if len(ring_atoms['in_furanose'])==5 and len(ring_atoms['in_pyranose'])==0:
                for atom_id in ring_atoms['in_furanose']:
                    if atom_id in res:
                        coord = res[atom_id].get_coord()
                        ch_coords.append(coord)

            elif len(ring_atoms['in_pyranose'])==6 and len(ring_atoms['in_furanose'])==0:
                for atom_id in ring_atoms['in_pyranose']:
                    if atom_id in res:
                        coord = res[atom_id].get_coord()
                        ch_coords.append(coord)
            else:
                continue

        

        if ch_coords:
            grid_size=half*2+1

            origin_grid_size=int(half/voxel_size.min())+4
            
            mass_center_random=np.mean(ch_coords,axis=0)+ (np.random.rand(3)*2-1)*voxel_size
            ch_mass_center_ind=np.round((mass_center_random- offset)/voxel_size).astype(int)
            if ch_mass_center_ind[0]-origin_grid_size<0 or ch_mass_center_ind[0]+origin_grid_size+1>=s0 or ch_mass_center_ind[1]-origin_grid_size<0 or ch_mass_center_ind[1]+origin_grid_size+1>=s1 or ch_mass_center_ind[2]-origin_grid_size<0 or ch_mass_center_ind[2]+origin_grid_size+1>=s2:
                continue
            sub_map_init=mrc_data[ch_mass_center_ind[0]-origin_grid_size:ch_mass_center_ind[0]+origin_grid_size+1,ch_mass_center_ind[1]-origin_grid_size:ch_mass_center_ind[1]+origin_grid_size+1,ch_mass_center_ind[2]-origin_grid_size:ch_mass_center_ind[2]+origin_grid_size+1]
            sub_map_init=torch.tensor(sub_map_init).view(1,1,origin_grid_size*2+1,origin_grid_size*2+1,origin_grid_size*2+1)
            # mass_center_random=ch_mass_center_ind*voxel_size
            # mass_center_random=ch_mass_center+ (np.random.rand(3)*2-1)*voxel_size


            mass_center_shift= mass_center_random-ch_mass_center_ind*voxel_size

            

            # 构建局部网格的索引（以中心为原点，步长为1）
            dz = torch.arange(-half, half+1)
            dy = torch.arange(-half, half+1)
            dx = torch.arange(-half, half+1)
            local_x, local_y, local_z = torch.meshgrid(dx, dy, dz, indexing='ij')  # (11,11,11)

            # 计算每个点的实际坐标
            # sample_z = (ch_mass_center_coord[0] + local_z - offset[0])/voxel_size[0]
            # sample_y = (ch_mass_center_coord[1] + local_y - offset[1])/voxel_size[1]
            # sample_x = (ch_mass_center_coord[2] + local_x - offset[2])/voxel_size[2]

            # 拼成 (11,11,11,3)
            sample_points = torch.stack([local_x, local_y, local_z], dim=-1)  # (11,11,11,3)
            sample_points=((sample_points+mass_center_shift))/(voxel_size*np.array([origin_grid_size,origin_grid_size,origin_grid_size]))
            grid=sample_points.view(1,grid_size,grid_size,grid_size,3)
            # import pdb
            # pdb.set_trace()
            if torch.sum(grid<-1)>0 or torch.sum(grid>1)>0:
                continue
            # print(sub_map_init.dtype,grid.dtype)
            sampled = grid_sample(sub_map_init, grid.float(), mode='bilinear', align_corners=True)  # (1,1,11,11,11)
            # print(sampled.mean())
            # print(sub_map_init.mean())
            
            
        
            atom_coord=[]
            atom_type=[]
            sampled_these=[]
            flag=True
            for atom in res.get_atoms():
                try:
                    ele=atom.element
                    if ele == 'H':
                        continue
                    coord = (atom.get_coord()-mass_center_random)/(voxel_size)#*np.array([origin_grid_size,origin_grid_size,origin_grid_size]))
                    cid=atom.get_id()
                    if cid in ring_atoms['in_furanose'] or cid in ring_atoms['in_pyranose']:
                        atom_coord.append(coord)
                        atom_type.append(atom_types['R'+ele])
                        coord_this=torch.tensor(coord/np.array([origin_grid_size,origin_grid_size,origin_grid_size])).view(1,1,1,1,3)
                        sampled_this=grid_sample(sampled, coord_this.float(), mode='bilinear', align_corners=True)
                        sampled_these.append(sampled_this)
                        # print(resname,sampled_this,sampled.mean())
                    else:
                        atom_coord.append(coord)
                        atom_type.append(atom_types[ele])
                except Exception as e:
                    flag=False
                    break
            
            
            if sampled_these:
                # val_list.append((np.mean(sampled_these) >=sampled.mean()))
                if np.mean(sampled_these) <sampled.mean():
                    continue
                    # import pdb
                    # pdb.set_trace()
            if flag:
                sub_map_list.append(SubMap(sampled[0,0],resname,atom_coord,atom_type))
            
            
            # sub_map_list.append(SubMap(sub_map_init,atom_coord,atom_type))
    # if val_list and np.mean(val_list)<1:
    #     print(pdbid,np.mean(val_list),np.sum(val_list))


    # print(len(start_x_list)*len(start_y_list)*len(start_z_list),len(sub_data_dict))
    if sub_map_list:
        result = ResultDiff(mrc_fn,cif_fn,offset,voxel_size,mrc_mean,mrc_max,p999,p99,origin_grid_size,sub_map_list)
        with open(os.path.join(training_dir,f'{pdbid}.pkl'), 'wb') as f:
            pickle.dump(result, f)
    return 'success'