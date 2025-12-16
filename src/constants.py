import json

chain_id_list=[]
for id in 'abcdefghijklmnopqrstuvwxyz':
    chain_id_list.append(id.upper())
for id in 'abcdefghijklmnopqrstuvwxyz':
    chain_id_list.append(id)
for id1 in 'abcdefghijklmnopqrstuvwxyz1234567890':
    for id2 in 'abcdefghijklmnopqrstuvwxyz1234567890':
        chain_id_list.append(id1+id2)

rings_cls={
    'non_atom':0,
    'atom_out_ring_4_7':1,
    'in_ring_4_7':2, # 排除以下类
    'in_fused_ring':3,  
    'in_furanose':4,
    'in_norm_na':5,
    'in_pyranose':6,
    'ch_mass_center':7
}
cls_rings={v:k for k,v in rings_cls.items()}

atom_cls={
    'non_atom':0,
    'atom_out_ring_4_7':1,
    'in_ring_4_7_box333':2,
    'in_ring_4_7':3
}
cls_atom={v:k for k,v in atom_cls.items()}
atom_cls_weight=[1.,20.,0.,100.]

ch_mass_cls={
    'non_atom':0,
    'in_ring_4_7_box333':1,
    'ch_mass_center':2,
}
cls_ch_mass={v:k for k,v in ch_mass_cls.items()}
ch_mass_cls_weight=[0.,10.,100.]

ch_cls={
    'non_atom':0,
    'in_furanose':1,
    'in_pyranose':2,
}
cls_ch={v:k for k,v in ch_cls.items()}
cls_ch_weight=[0.,1.,1.]


na_cls={
    'G':0,
    'A':1,
    'C':2,
    'U':3,
    'DA':4,
    'DG':5,
    'DC':6,
    'DT':7,
}
cls_na={v:k for k,v in na_cls.items()}

with open('data/json/ligand.json','r') as f:
    ligand_rings=json.load(f)

atom_types={
    'RC':0,
    'RO':1,
    'C':2,
    'O':3,
    'N':4,
    'P':5,
    'S':6,
    '-':7,
}
atom_types_cls={v:k for k,v in atom_types.items()}

atom_charges={
    'RC':6,
    'RO':8,
    'C':6,
    'O':8,
    'N':7,
    'P':15,
    'S':16,
    '-':0,
}




all_standard_chs=['BGC','GLC','GAL','GLA','GZL','BMA','MAN','AHR','FUB','ARA','ARB','XYP','XYS','FUC','FUL','RAM','RM4','NAG','NDG','A2G','NGA']


# with open('data/json/ele2label.json', "r", encoding="utf-8") as f:
#     ele2label = json.load(f)

# with open('data/json/res2label.json', "r", encoding="utf-8") as f:
#     res2label = json.load(f)

# with open('data/json/atom2label.json', "r", encoding="utf-8") as f:
#     atom2label = json.load(f)

