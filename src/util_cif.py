import os
import numpy as np
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.MMCIFParser import MMCIFParser

from openbabel import pybel

from Bio.PDB import MMCIFIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain

import gemmi


def save_residue_as_cif(residue, output_file):
    """
    将单个残基保存为CIF文件
    
    Args:
        residue: Bio.PDB.Residue对象
        output_file: 输出文件路径
    """
    # 创建结构层次
    structure = Structure("structure")
    model = Model(0)
    chain = Chain("A")
    
    # 添加残基到链中
    chain.add(residue)
    model.add(chain)
    structure.add(model)
    
    # 保存为CIF文件
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(output_file)

def extract_residue_from_chem_comp_cif(cif_file, res_id=(' ', 1, ' ')):
    doc = gemmi.cif.read(cif_file)
    with open(cif_file,'r') as f:
        lines=f.readlines()
        bond_lines=[]
        flag=False
        for idx, line in enumerate(lines):
            bond_lines.append(line)
            if line.startswith('_chem_comp_bond'):
                flag=True
            if line.startswith('#') or idx>=len(lines)-1:
                if flag:
                    break
                bond_lines=[]
    for block in doc:
        comp_ids = block.find_loop('_chem_comp_atom.comp_id')
        if not comp_ids:
            continue
        atom_ids = block.find_loop('_chem_comp_atom.atom_id')
        alt_atom_ids = block.find_loop('_chem_comp_atom.alt_atom_id')
        type_symbols = block.find_loop('_chem_comp_atom.type_symbol')
        # charges = block.find_loop('_chem_comp_atom.charge')
        # pdbx_aligns = block.find_loop('_chem_comp_atom.pdbx_align')
        # pdbx_aromatic_flags = block.find_loop('_chem_comp_atom.pdbx_aromatic_flag')
        # pdbx_leaving_atom_flags = block.find_loop('_chem_comp_atom.pdbx_leaving_atom_flag')
        # pdbx_stereo_configs = block.find_loop('_chem_comp_atom.pdbx_stereo_config')
        # pdbx_backbone_atom_flags = block.find_loop('_chem_comp_atom.pdbx_backbone_atom_flag')
        # pdbx_n_terminal_atom_flags = block.find_loop('_chem_comp_atom.pdbx_n_terminal_atom_flag')
        # pdbx_c_terminal_atom_flags = block.find_loop('_chem_comp_atom.pdbx_c_terminal_atom_flag')
        model_Cartn_xs = block.find_loop('_chem_comp_atom.model_Cartn_x')
        model_Cartn_ys = block.find_loop('_chem_comp_atom.model_Cartn_y')
        model_Cartn_zs = block.find_loop('_chem_comp_atom.model_Cartn_z')
        pdbx_model_Cartn_x_ideals = block.find_loop('_chem_comp_atom.pdbx_model_Cartn_x_ideal')
        pdbx_model_Cartn_y_ideals = block.find_loop('_chem_comp_atom.pdbx_model_Cartn_y_ideal')
        pdbx_model_Cartn_z_ideals = block.find_loop('_chem_comp_atom.pdbx_model_Cartn_z_ideal')
        # pdbx_component_atom_ids = block.find_loop('_chem_comp_atom.pdbx_component_atom_id')
        # pdbx_component_comp_ids = block.find_loop('_chem_comp_atom.pdbx_component_comp_id')
        pdbx_ordinals = block.find_loop('_chem_comp_atom.pdbx_ordinal')
        residue = Residue((' ', 1, ' '), comp_ids[0], '')
        residue.bonds = []
        row_idx=0
        # for atom_id,alt_atom_id,type_symbol,charge,pdbx_align,pdbx_aromatic_flag,pdbx_leaving_atom_flag,pdbx_stereo_config,pdbx_backbone_atom_flag,pdbx_n_terminal_atom_flag,pdbx_c_terminal_atom_flag,model_Cartn_x,model_Cartn_y,model_Cartn_z,pdbx_model_Cartn_x_ideal,pdbx_model_Cartn_y_ideal,pdbx_model_Cartn_z_ideal,pdbx_component_atom_id,pdbx_component_comp_id,pdbx_ordinal in zip(atom_ids,alt_atom_ids,type_symbols,charges,pdbx_aligns,pdbx_aromatic_flags,pdbx_leaving_atom_flags,pdbx_stereo_configs,pdbx_backbone_atom_flags,pdbx_n_terminal_atom_flags,pdbx_c_terminal_atom_flags,model_Cartn_xs,model_Cartn_ys,model_Cartn_zs,pdbx_model_Cartn_x_ideals,pdbx_model_Cartn_y_ideals,pdbx_model_Cartn_z_ideals,pdbx_component_atom_ids,pdbx_component_comp_ids,pdbx_ordinals):
        x_list=[]
        atom_id_list=[]
        for atom_id,alt_atom_id,type_symbol,model_Cartn_x,model_Cartn_y,model_Cartn_z,pdbx_model_Cartn_x_ideal,pdbx_model_Cartn_y_ideal,pdbx_model_Cartn_z_ideal,pdbx_ordinal in zip(atom_ids,alt_atom_ids,type_symbols,model_Cartn_xs,model_Cartn_ys,model_Cartn_zs,pdbx_model_Cartn_x_ideals,pdbx_model_Cartn_y_ideals,pdbx_model_Cartn_z_ideals,pdbx_ordinals):
            while atom_id[0]==atom_id[-1]=='"' or atom_id[0]==atom_id[-1]=="'":
                atom_id=atom_id[1:-1]
            atom_id_list.append(atom_id)
            row_idx+=1
            try:
                x=float(pdbx_model_Cartn_x_ideal)
                y=float(pdbx_model_Cartn_y_ideal)
                z=float(pdbx_model_Cartn_z_ideal)
            except:
                try:
                    x=float(model_Cartn_x)
                    y=float(model_Cartn_y)
                    z=float(model_Cartn_z)
                except:
                    return None, None, None, None
            x_list.append(x)

            atom_name=atom_id if atom_id else alt_atom_id
            element=type_symbol
            coord = [x, y, z]
            atom = Atom(
                name=atom_name,
                coord=coord,
                element=element,
                bfactor=20.0,  # 默认B因子
                occupancy=1.0,  # 默认占有率
                altloc=' ',
                fullname=atom_name,
                serial_number=pdbx_ordinal
            )
            residue.add(atom)

        bond_comp_ids = block.find_loop('_chem_comp_bond.comp_id')
        bond_atom_id_1s = block.find_loop('_chem_comp_bond.atom_id_1')
        bond_atom_id_2s = block.find_loop('_chem_comp_bond.atom_id_2')
        bond_value_orders = block.find_loop('_chem_comp_bond.value_order')
        bond_aromatic_flags = block.find_loop('_chem_comp_bond.pdbx_aromatic_flag')
        bond_stereo_configs = block.find_loop('_chem_comp_bond.pdbx_stereo_config')
        bond_ordinals = block.find_loop('_chem_comp_bond.pdbx_ordinal')
        for bond_comp_id,bond_atom_id_1,bond_atom_id_2,bond_value_order,bond_aromatic_flag,bond_stereo_config,bond_ordinal in zip(bond_comp_ids,bond_atom_id_1s,bond_atom_id_2s,bond_value_orders,bond_aromatic_flags,bond_stereo_configs,bond_ordinals):
            residue.bonds.append({
                'comp_id': bond_comp_id,
                'atom_id_1': bond_atom_id_1,
                'atom_id_2': bond_atom_id_2,
                'value_order': bond_value_order,
                'aromatic_flag': bond_aromatic_flag,
                'stereo_config': bond_stereo_config,
                'ordinal': bond_ordinal
            })
        return residue, atom_id_list, x_list, bond_lines
    return None, None, None, None

