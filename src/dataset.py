import mrcfile
import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import sys
import pickle
from random import random
from scipy.ndimage import convolve
from .constants import rings_cls,atom_cls,ch_mass_cls,ch_cls
sys.path.append('..')

class GridData(Dataset):
    def __init__(self,data_file,data_dir,data_partion,input_norm,data_argu,train,test,finetune,finetune_emid,task='ring_cls'):
        self.task=task
        self.data_partion=data_partion
        self.data_dir=data_dir
        self.inputs=[]
        self.label_raws=[]
        self.label_atoms=[]

        df= pd.read_csv(data_file,dtype=str)
        df=df[::data_partion]
        pbar=tqdm(zip(df['emid'],df['pdbid']),total=len(df))
        py_all=0
        fur_all=0
        for emname,pdbid in pbar:
            pkl_path=os.path.join(data_dir,f'{pdbid}.pkl')
            try:
                with open(pkl_path,'rb') as rb:
                    meta_data=pickle.load(rb)
            except:
                continue
            label_data=meta_data.data
            mrc_fn=meta_data.mrc_fn
            p99=meta_data.p99

            mrc_data=np.transpose(mrcfile.open(mrc_fn).data.copy(), [2,1,0])
            if mrc_data.shape[0]<64 or mrc_data.shape[1]<64 or mrc_data.shape[2]<64:
                mrc_data = np.pad(mrc_data, [(0, max(0,64-mrc_data.shape[0])), (0, max(0,64-mrc_data.shape[1])), (0, max(0,64-mrc_data.shape[2]))], 'constant', constant_values=[(0, 0), (0, 0), (0, 0)])
            mrc_data=mrc_data/p99
            mrc_data=mrc_data.astype(np.float16)

            na_flag=False
            for ind in label_data:
                if task=='atom_cls':
                    sub_raw_cls=label_data[ind].raw_cls
                    num=np.sum(sub_raw_cls>=rings_cls['atom_out_ring_4_7'])
                    if random()<(num+1)/100:
                        sub_atom_cls=label_data[ind].label_atom
                        sub_mrc=mrc_data[ind[0]:ind[0]+64,ind[1]:ind[1]+64,ind[2]:ind[2]+64]
                        sub_atom_cls[sub_raw_cls>=rings_cls['atom_out_ring_4_7']] = atom_cls['atom_out_ring_4_7']
                        sub_mrc = np.expand_dims(sub_mrc, axis=0)
                        self.inputs.append(sub_mrc)
                        # self.label_raws.append(sub_raw_cls)
                        self.label_atoms.append(sub_atom_cls)
                elif task=='ring_cls':
                    sub_raw_cls=label_data[ind].raw_cls
                    num=np.sum(sub_raw_cls>=rings_cls['atom_out_ring_4_7'])
                    if random()<(num+1)/100:
                        sub_atom_cls=label_data[ind].label_atom
                        sub_mrc=mrc_data[ind[0]:ind[0]+64,ind[1]:ind[1]+64,ind[2]:ind[2]+64]
                        sub_atom_cls[sub_raw_cls>=rings_cls['in_ring_4_7']] = atom_cls['in_ring_4_7']
                        sub_mrc = np.expand_dims(sub_mrc, axis=0)
                        self.inputs.append(sub_mrc)
                        # self.label_raws.append(sub_raw_cls)
                        self.label_atoms.append(sub_atom_cls)
                elif task=='find_ch':
                    sub_atom_cls=label_data[ind].label_atom
                    num=np.sum(sub_atom_cls>=atom_cls['in_ring_4_7_box333'])
                    if random()<(num+1)/300:
                        sub_raw_cls=label_data[ind].raw_cls
                        sub_atom_cls[sub_atom_cls==atom_cls['atom_out_ring_4_7']] = ch_mass_cls['non_atom'] # 0
                        sub_atom_cls[sub_atom_cls>=atom_cls['in_ring_4_7_box333']] = ch_mass_cls['in_ring_4_7_box333'] # 1
                        sub_mrc=mrc_data[ind[0]:ind[0]+64,ind[1]:ind[1]+64,ind[2]:ind[2]+64]
                        sub_atom_cls[sub_raw_cls>=rings_cls['ch_mass_center']] = ch_mass_cls['ch_mass_center'] # 2
                        sub_mrc = np.expand_dims(sub_mrc, axis=0)
                        self.inputs.append(sub_mrc)
                        # self.label_raws.append(sub_raw_cls)
                        self.label_atoms.append(sub_atom_cls)
                elif task=='cls_ch':
                    sub_raw_cls=label_data[ind].raw_cls
                    num=np.sum(sub_raw_cls>=rings_cls['in_furanose'])
                    if random()<(num+1)/30:
                        sub_atom_cls=label_data[ind].label_atom
                        sub_atom_cls[sub_raw_cls<=rings_cls['in_fused_ring']] = ch_cls['non_atom'] # 0
                        fur_ind=(sub_raw_cls>=rings_cls['in_furanose'])*(sub_raw_cls<=rings_cls['in_norm_na'])
                        fur_count=convolve(fur_ind,np.ones((3,3,3)),mode='constant')
                        py_count=convolve((sub_raw_cls==rings_cls['in_pyranose']),np.ones((3,3,3)),mode='constant')
                        sub_atom_cls[py_count<fur_count] = ch_cls['in_furanose'] # 1
                        sub_atom_cls[py_count>fur_count] = ch_cls['in_pyranose'] # 2
                        sub_mrc=mrc_data[ind[0]:ind[0]+64,ind[1]:ind[1]+64,ind[2]:ind[2]+64]
                        sub_mrc = np.expand_dims(sub_mrc, axis=0)
                        
                        # if np.sum(sub_raw_cls==rings_cls['in_norm_na']) > 0 and fur_all> py_all:
                        #     continue
                        if np.sum(sub_raw_cls==rings_cls['in_norm_na']) > 10 and np.sum(sub_raw_cls==rings_cls['in_pyranose'])==0:
                            if na_flag:
                                continue
                            na_flag=True
                        
                        fur_this=np.sum(sub_atom_cls==ch_cls['in_furanose'])
                        py_this=np.sum(sub_atom_cls==ch_cls['in_pyranose'])
                        fur_all+=fur_this # 1
                        py_all+=py_this # 2
                        # print(len(self.inputs),fur_all, py_all)
                        
                        self.inputs.append(sub_mrc)
                        # self.label_raws.append(sub_raw_cls)
                        self.label_atoms.append(sub_atom_cls)
            del meta_data,mrc_data

        print(f'data num:{len(self.inputs)}')

    def __getitem__(self, index):
        # if self.task=='ring_cls':
        #     return self.inputs[index],self.label_atoms[index].astype('long')
        # elif self.task=='find_ch':
        #     return self.inputs[index],self.label_atoms[index].astype('long')
        # elif self.task=='cls_ch':
            return self.inputs[index],self.label_atoms[index].astype('long')

    def __len__(self):
        return len(self.inputs)

class GridDataModule(L.LightningDataModule):
    def __init__(self, train_csv: str, val_csv: str, train_dir: str,val_dir : str, batch_size: int, val_batch_size: int, data_partion:int, input_norm:str ,data_argu:int,num_worker:int, val_data_partion=1,finetune=False, finetune_emid='', test=False, task='aa_cls'):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size=val_batch_size
        self.train_dataset=GridData(data_file=train_csv,data_dir=train_dir,data_partion=data_partion,input_norm=input_norm,data_argu=data_argu,finetune=finetune,finetune_emid=finetune_emid,task=task,train=True,test=False)
        self.val_dataset=GridData(data_file=val_csv,data_dir=val_dir,data_partion=val_data_partion,input_norm=input_norm,data_argu=data_argu,finetune=finetune,finetune_emid=finetune_emid,task=task,train=False,test=False)
        self.test=test
        if test:
            self.test_dataset=GridData(data_file=val_csv,data_dir=val_dir,data_partion=1,input_norm=input_norm,data_argu=data_argu,finetune=finetune,finetune_emid=finetune_emid,task=task,train=False,test=True)
            self.test_sampler = SequentialSampler(self.test_dataset)
        self.train_sampler = RandomSampler(self.train_dataset)

        self.val_sampler = SequentialSampler(self.val_dataset)
        
        self.num_worker=num_worker

        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, sampler=self.train_sampler, batch_size=self.batch_size, num_workers=self.num_worker)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, sampler=self.val_sampler, batch_size=self.val_batch_size, num_workers=self.num_worker)
    
    def test_dataloader(self):
        # if self.test:
            return DataLoader(self.test_dataset, sampler=self.test_sampler, batch_size=1, num_workers=1)
        # else:
        #     return super().test_dataloader()