# EModelG

**For GF-NP5 with bond**:   
python auto_build.py   --task build_bond   --origin_map data/GF-NP5/map_P33-J1422.mrc   --map_fn data/GF-NP5/NP5_zone_A_3.mrc   --o_link_pdb data/GF-NP5/map_P33-J1422-coot-2_real_space_refined_212.pdb   --o_link_site "('A',('H_LIG', 1, ' '))"  --cand_num 2000 --resol 2.22    

**For GF-NP5 w/o bond**:   
python auto_build.py   --task no_bond  --map_fn data/GF-NP5/map_P33-J1422.mrc --cand_num 20000 --resol 2.22   

**Find your results**:   
data/GF-NP5/final_NP5_zone_A_3_conn_with_type_real_space_refined_000.cif   
data/GF-NP5/final_map_P33-J1422_initial_real_space_refined_000.cif