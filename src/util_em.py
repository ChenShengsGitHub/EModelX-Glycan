import os
import pandas as pd
import numpy as np
import mrcfile as mrc
from copy import deepcopy

from collections import namedtuple

from Bio.PDB import MMCIFParser
cif_parser=MMCIFParser(QUIET=True)

MRCObject = namedtuple("MRCObject", ["grid", "voxel_size", "global_origin","mapc","mapr","maps"])


def normalize_map(map_fn,cif_fn,mrc_fn,target_voxel_size):
    mrc_file = mrc.open(map_fn, "r")
    cellb = mrc_file.header["cellb"]
    allowed_chars = set(' 90().,')
    if not set(str(cellb)).issubset(allowed_chars):
        raise RuntimeError(f"{map_fn} cellb string contains invalid characters: {str(cellb)}")
    
    voxel_size = mrc_file.voxel_size
    if voxel_size.x <=0 or voxel_size.y <=0 or voxel_size.z <= 0:
        raise RuntimeError(f"{map_fn} voxel_size error {voxel_size.x},{voxel_size.y},{voxel_size.z}.")

    
    c = mrc_file.header["mapc"] # 哪一维是z
    r = mrc_file.header["mapr"] # 哪一维是y
    s = mrc_file.header["maps"] # 哪一维是x

    ori_origin = mrc_file.header["origin"]
    ori_origin = deepcopy(ori_origin)

    ncstart=mrc_file.header["nxstart"]
    nrstart=mrc_file.header["nystart"]
    nsstart=mrc_file.header["nzstart"]

    grid = np.copy(mrc_file.data)
    if c == 1 and r == 2 and s == 3:
        perm_xyz=[0,1,2]
        perm_zyx=[2,1,0]
    elif c == 3 and r == 2 and s == 1:
        grid = np.moveaxis(grid, [0, 1, 2], [2, 1, 0])
        perm_xyz=[2,1,0]
        perm_zyx=[0,1,2]
    elif c == 2 and r == 1 and s == 3:
        grid = np.moveaxis(grid, [1, 2, 0], [2, 1, 0])
        perm_xyz=[0,2,1]
        perm_zyx=[1,2,0]
    else:
        raise RuntimeError("MRC file axis arrangement not supported!")
    start_xyz=np.array([ncstart,nrstart,nsstart])[perm_xyz]
    voxel_size_xyz=np.array([voxel_size.x,voxel_size.y,voxel_size.z])[perm_xyz]
    start_zyx=np.array([ncstart,nrstart,nsstart])[perm_zyx]
    voxel_size_zyx=np.array([voxel_size.x,voxel_size.y,voxel_size.z])[perm_zyx]

    origin_start_xyz=np.array([ori_origin.x,ori_origin.y,ori_origin.z])+start_xyz*voxel_size_xyz
    # print(voxel_size_xyz,start_xyz)

    
    if cif_fn is not None:
        gt_structure=cif_parser.get_structure('',cif_fn)
        gt_model=gt_structure[0]
        gt_atoms_coords_ori=np.array([atom.get_coord() for atom in gt_model.get_atoms() if atom.element!='H'])

    if cif_fn is not None:
        gt_atoms_coords_int=np.round((gt_atoms_coords_ori-origin_start_xyz)/voxel_size_xyz).astype(int)
        gt_atoms_coords_int=gt_atoms_coords_int[gt_atoms_coords_int[:,2]>=0,:]
        gt_atoms_coords_int=gt_atoms_coords_int[gt_atoms_coords_int[:,2]<grid.shape[0],:]
        gt_atoms_coords_int=gt_atoms_coords_int[gt_atoms_coords_int[:,1]>=0,:]
        gt_atoms_coords_int=gt_atoms_coords_int[gt_atoms_coords_int[:,1]<grid.shape[1],:]
        gt_atoms_coords_int=gt_atoms_coords_int[gt_atoms_coords_int[:,0]>=0,:]
        gt_atoms_coords_int=gt_atoms_coords_int[gt_atoms_coords_int[:,0]<grid.shape[2],:]
        # print(mrc_file.header["origin"],mrc_file.header["nxstart"],mrc_file.header["nystart"],mrc_file.header["nzstart"],voxel_size)
        if gt_atoms_coords_int.shape[0]<=0:
            raise RuntimeError(f'{cif_fn}: {grid.shape}, {c}, {r}, {s}  out of box')
        else:
            mean_dens=np.mean(grid)
            over_score1=np.sum(grid[gt_atoms_coords_int[:,2],gt_atoms_coords_int[:,1],gt_atoms_coords_int[:,0]]>mean_dens)/gt_atoms_coords_ori.shape[0]
            # print(f'{cif_fn}: {grid.shape}, {mrc_file.header["nxstart"]},{mrc_file.header["nystart"]},{mrc_file.header["nzstart"]},{c}, {r}, {s} {over_score}')

    grid = grid - np.mean(grid)
    grid, shift, _ = make_cubic(grid)
    origin_start_xyz -= shift[::-1]* voxel_size_xyz
    grid, voxel_size_new = normalize_voxel_size( # bug
        grid, voxel_size_zyx, target_voxel_size=target_voxel_size
    )

    if cif_fn is not None:
        gt_atoms_coords=np.array([atom.get_coord() for atom in gt_model.get_atoms()])
        gt_atoms_coords=(gt_atoms_coords-origin_start_xyz)/voxel_size_new[::-1]
        # 计算gt_atoms_coords的边界
        min_coords = np.floor(gt_atoms_coords.min(axis=0)).astype(int)
        max_coords = np.ceil(gt_atoms_coords.max(axis=0)).astype(int)

        # 向外拓展10个voxel
        expand = 10
        grid_shape = grid.shape

        xmin = max(min_coords[0] - expand, 0)
        ymin = max(min_coords[1] - expand, 0)
        zmin = max(min_coords[2] - expand, 0)

        xmax = max_coords[0] + expand
        ymax = max_coords[1] + expand
        zmax = max_coords[2] + expand
        
        # 剪裁grid
        
        box_min=np.array([xmin,ymin,zmin])
        box_max=np.array([xmax,ymax,zmax])
        grid = grid[box_min[2]:min(box_max[2]+1,grid_shape[0]), box_min[1]:min(box_max[1]+1,grid_shape[1]), box_min[0]:min(box_max[0]+1,grid_shape[2])]
        origin_start_xyz+=box_min*voxel_size_new[::-1]
        
        gt_atoms_coords_int=np.round((gt_atoms_coords_ori-origin_start_xyz)/voxel_size_new[::-1]).astype(int)
        gt_atoms_coords_int=gt_atoms_coords_int[gt_atoms_coords_int[:,2]>=0,:]
        gt_atoms_coords_int=gt_atoms_coords_int[gt_atoms_coords_int[:,2]<grid.shape[0],:]
        gt_atoms_coords_int=gt_atoms_coords_int[gt_atoms_coords_int[:,1]>=0,:]
        gt_atoms_coords_int=gt_atoms_coords_int[gt_atoms_coords_int[:,1]<grid.shape[1],:]
        gt_atoms_coords_int=gt_atoms_coords_int[gt_atoms_coords_int[:,0]>=0,:]
        gt_atoms_coords_int=gt_atoms_coords_int[gt_atoms_coords_int[:,0]<grid.shape[2],:]
        # print(mrc_file.header["origin"],mrc_file.header["nxstart"],mrc_file.header["nystart"],mrc_file.header["nzstart"],voxel_size)
        if gt_atoms_coords_int.shape[0]<=0:
            raise RuntimeError(f'{cif_fn}: {grid.shape}, {c}, {r}, {s}  out of box')
        else:
            mean_dens=np.mean(grid)
            over_score2=np.sum(grid[gt_atoms_coords_int[:,2],gt_atoms_coords_int[:,1],gt_atoms_coords_int[:,0]]>mean_dens)/gt_atoms_coords_ori.shape[0]
            # print(f'{cif_fn}: {grid.shape}, {mrc_file.header["nxstart"]},{mrc_file.header["nystart"]},{mrc_file.header["nzstart"]},{c}, {r}, {s} {over_score}')
        

    (nz, ny, nx) = grid.shape
    o = mrc.new(mrc_fn, overwrite=True)
    o.header["cella"].x = nx * voxel_size_new[0]
    o.header["cella"].y = ny * voxel_size_new[1]
    o.header["cella"].z = nz * voxel_size_new[2]

    o.header["origin"].x = origin_start_xyz[0]
    o.header["origin"].y = origin_start_xyz[1]
    o.header["origin"].z = origin_start_xyz[2]
    o.set_data(grid.astype(np.float32))
    o.update_header_stats()
    o.flush()
    o.close()
    if cif_fn is not None:
        return origin_start_xyz, grid.shape, voxel_size_new, over_score1,over_score2
    return origin_start_xyz, grid.shape, voxel_size_new, 0, 0

def make_cubic(box):
    bz = np.array(box.shape)
    s = np.max(box.shape)
    s = max(s, 128)
    s += s % 2
    if np.all(box.shape == s):
        return box, np.zeros(3, dtype=np.int64), bz
    nbox = np.zeros((s, s, s))
    c = np.array(nbox.shape) // 2 - bz // 2
    nbox[c[0] : c[0] + bz[0], c[1] : c[1] + bz[1], c[2] : c[2] + bz[2]] = box
    return nbox, c, c + bz

def normalize_voxel_size(density, in_voxel_sz_list, target_voxel_size=1.0):
    (iz, iy, ix) = np.shape(density)

    assert iz % 2 == 0 and iy % 2 == 0 and ix % 2 == 0
    assert ix == iy == iz

    in_sz = ix
    out_sz = int(round(in_sz * in_voxel_sz_list[0] / target_voxel_size))
    if out_sz % 2 != 0:
        vs1 = in_voxel_sz_list[0] * in_sz / (out_sz + 1)
        vs2 = in_voxel_sz_list[0] * in_sz / (out_sz - 1)
        if np.abs(vs1 - target_voxel_size) < np.abs(vs2 - target_voxel_size):
            out_sz += 1
        else:
            out_sz -= 1

    out_voxel_sz_list=deepcopy(in_voxel_sz_list) # bug
    out_voxel_sz_list[0] = in_voxel_sz_list[0] * in_sz / out_sz
    out_voxel_sz_list[1] = in_voxel_sz_list[1] * in_sz / out_sz
    out_voxel_sz_list[2] = in_voxel_sz_list[2] * in_sz / out_sz
    density = rescale_real(density, out_sz)

    return density, out_voxel_sz_list

def rescale_real(box, out_sz):
    if out_sz != box.shape[0]:
        f = np.fft.rfftn(box)
        f = rescale_fourier(f, out_sz)
        box = np.fft.irfftn(f)

    return box

def rescale_fourier(box, out_sz):
    if out_sz % 2 != 0:
        raise Exception("Bad output size")
    if box.shape[0] != box.shape[1] or box.shape[1] != (box.shape[2] - 1) * 2:
        raise Exception("Input must be cubic")

    ibox = np.fft.ifftshift(box, axes=(0, 1))
    obox = np.zeros((out_sz, out_sz, out_sz // 2 + 1), dtype=box.dtype)

    si = np.array(ibox.shape) // 2
    so = np.array(obox.shape) // 2

    if so[0] < si[0]:
        obox = ibox[
            si[0] - so[0] : si[0] + so[0],
            si[1] - so[1] : si[1] + so[1],
            : obox.shape[2],
        ]
    elif so[0] > si[0]:
        obox[
            so[0] - si[0] : so[0] + si[0],
            so[1] - si[1] : so[1] + si[1],
            : ibox.shape[2],
        ] = ibox
    else:
        obox = ibox

    obox = np.fft.ifftshift(obox, axes=(0, 1))

    return obox


def save_mrc(mrc_obj, filename,cif_fn=None):
    grid, voxel_size, origin,c,r,s=mrc_obj
    if c == 3 and r == 2 and s == 1:
        temp=origin.x
        origin.x=origin.z
        origin.z=temp
    elif c == 2 and r == 1 and s == 3:
        temp=origin.x
        origin.x=origin.y
        origin.y=temp
    if cif_fn is not None:
        gt_structure=cif_parser.get_structure('',cif_fn)
        gt_model=gt_structure[0]
        gt_atoms_coords_ori=np.array([atom.get_coord() for atom in gt_model.get_atoms()])
        gt_atoms_coords=(gt_atoms_coords_ori-np.array([origin.x,origin.y,origin.z]))/np.array([voxel_size.x,voxel_size.y,voxel_size.z])
        # 计算gt_atoms_coords的边界
        min_coords = np.floor(gt_atoms_coords.min(axis=0)).astype(int)
        max_coords = np.ceil(gt_atoms_coords.max(axis=0)).astype(int)

        # 向外拓展10个voxel
        expand = 10
        grid_shape = grid.shape

        xmin = max(min_coords[0] - expand, 0)
        ymin = max(min_coords[1] - expand, 0)
        zmin = max(min_coords[2] - expand, 0)

        xmax = max_coords[0] + expand
        ymax = max_coords[1] + expand
        zmax = max_coords[2] + expand
        
        # 剪裁grid
        
        box_min=np.array([xmin,ymin,zmin])
        box_max=np.array([xmax,ymax,zmax])

        # # 电镜图像的axis顺序通常由c, r, s决定（比如MRC文件的axis order），
        # # 这里需要根据c, r, s对box_min和box_max进行转置，使其与grid的实际轴顺序一致
        # # 下面的代码实现了根据crs对box_min和box_max的转置
        # # 例如，如果c=3, r=2, s=1，表示grid的第0轴对应z(3)，第1轴对应y(2)，第2轴对应x(1)
        # # 所以需要把box_min/max的顺序从[z, y, x] -> [x, y, z]
        # # 更通用的写法如下：
        # crs = [c, r, s]
        # # crs的值是1,2,3的排列，代表xyz
        # # 需要将box_min/max的顺序从[z, y, x]映射到[axis0, axis1, axis2]
        # # 先构造一个perm数组，perm[i]表示grid的第i轴对应box_min/max的哪个分量
        # # 例如crs=[3,2,1]，则perm=[2,1,0]
        # perfect_match=[1,2,3]
        # perm = [crs.index(perfect_match[i]) for i in range(3)]
        # box_min = box_min[perm]
        # box_max = box_max[perm]
        # print(grid.shape, end=' ')
        grid = grid[box_min[2]:min(box_max[2]+1,grid_shape[0]), box_min[1]:min(box_max[1]+1,grid_shape[1]), box_min[0]:min(box_max[0]+1,grid_shape[2])]
        # print(grid.shape, box_min, box_max, c, r, s)
        
        # 根据剪裁的起点调整global_origin
        origin.x += box_min[0] * voxel_size.x
        origin.y += box_min[1] * voxel_size.y
        origin.z += box_min[2] * voxel_size.z
        gt_atoms_coords_int=(np.round(gt_atoms_coords_ori-np.array([origin.x,origin.y,origin.z]))/np.array([voxel_size.x,voxel_size.y,voxel_size.z])).astype(int)
        gt_atoms_coords_int=gt_atoms_coords_int[gt_atoms_coords_int[:,2]>=0,:]
        gt_atoms_coords_int=gt_atoms_coords_int[gt_atoms_coords_int[:,2]<grid.shape[0],:]
        gt_atoms_coords_int=gt_atoms_coords_int[gt_atoms_coords_int[:,1]>=0,:]
        gt_atoms_coords_int=gt_atoms_coords_int[gt_atoms_coords_int[:,1]<grid.shape[1],:]
        gt_atoms_coords_int=gt_atoms_coords_int[gt_atoms_coords_int[:,0]>=0,:]
        gt_atoms_coords_int=gt_atoms_coords_int[gt_atoms_coords_int[:,0]<grid.shape[2],:]
        if gt_atoms_coords_int.shape[0]<=gt_atoms_coords_ori.shape[0]/2:
            print(f'{cif_fn}: {grid.shape}, {c}, {r}, {s}  has {gt_atoms_coords_int.shape[0]} atoms, which is less than half of the original {gt_atoms_coords_ori.shape[0]} atoms')
        else:
            mean_dens=np.mean(grid)
            std_dens=np.std(grid)
            mean_coord_dens=np.mean(grid[gt_atoms_coords_int[:,2],gt_atoms_coords_int[:,1],gt_atoms_coords_int[:,0]])
            # if mean_coord_dens<mean_dens+std_dens:
            print(f'{cif_fn}: {grid.shape}, {c}, {r}, {s} {mean_coord_dens},{mean_dens},{std_dens}')

    (z, y, x) = grid.shape
    o = mrc.new(filename, overwrite=True)
    o.header["cella"].x = x * voxel_size.x
    o.header["cella"].y = y * voxel_size.y
    o.header["cella"].z = z * voxel_size.z
    # o._set_voxel_size((voxel_size,voxel_size,voxel_size))
    
    o.header["origin"].x = origin.x
    o.header["origin"].y = origin.y
    o.header["origin"].z = origin.z

    # o.header["nxstart"] = origin.x
    # o.header["nystart"] = origin.y
    # o.header["nzstart"] = origin.z
    # o.header["nxstart"] = origin.x/voxel_size
    # o.header["nystart"] = origin.y/voxel_size
    # o.header["nzstart"] = origin.z/voxel_size
    # print(o.header.nxstart,o.header.nystart,o.header.nzstart)
    # out_box = np.reshape(grid, (z, y, x))
    o.set_data(grid.astype(np.float32))
    # o.header["cella"].x=1000
    o.update_header_stats()
    o.flush()
    # print(o.voxel_size)
    # o.print_header()
    o.close()
    return origin, grid.shape, voxel_size