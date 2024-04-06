#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import h5py
import glob
import csv
from tqdm import tqdm

def read_pts(filename):
    return np.loadtxt(filename)

def read_label(filename):
    return np.loadtxt(filename)

def load_data(train, validation, pcd_size=2907):
    if ((train== True) and (validation == False)):
        partition = 'train'
    elif((train== True) and (validation == True)):
        partition = 'val'
    else:
        partition = 'test'
    BASE_DIR = os.path.join("../", "../")
    DATA_DIR = os.path.join(BASE_DIR, partition, 'pts')
    LABEL_DIR = os.path.join(BASE_DIR, partition, 'label')
    all_data = []
    all_label = []
    all_part_id = []
    for _, _, fs in os.walk(DATA_DIR):
        for f in tqdm(fs):
            file_name = f[:-4]
            label_path = os.path.join(LABEL_DIR, file_name+".txt")
            pcd_path = os.path.join(DATA_DIR, f)
            # load pcd and label data
            data = read_pts(pcd_path)
            part_id = read_label(label_path)

            # pad for alignment
            pad_size = pcd_size - data.shape[0]
            data = np.pad(data, ((pad_size//2, pad_size - (pad_size//2)), (0, 0)), 'edge')
            part_id = np.pad(part_id, ((pad_size//2, pad_size - (pad_size//2))), 'edge')
            label = np.ones(part_id.shape)

            # update the complete dataset
            all_data.append(data[None, ...])
            all_label.append(label[None, ...])
            all_part_id.append(part_id[None, ...])
    all_data = torch.from_numpy(np.concatenate(all_data, axis = 0))
    all_label = torch.from_numpy(np.concatenate(all_label, axis= 0))
    all_part_id = torch.from_numpy(np.concatenate(all_part_id, axis= 0))
    print(all_data.shape)
    return all_data,all_label,all_part_id    

#a,b,c = load_data(train = False, validation = False)
#print(c.size())

class ShapeNetData(Dataset):
    def __init__(self, train = True, validation = False, num_points = 2907, randomize_data = False):
        super(ShapeNetData, self).__init__()
        self.data, self.labels, self.part_id = load_data(train, validation, pcd_size=num_points)
        self.num_points = num_points
        self.randomize_data = randomize_data
        # if not train:
        #     self.shapes = self.read_classes_ShapeNet()
            
    def __getitem__(self, idx):
        # if self.randomize_data:
        #     current_points = self.randomize(idx)
        # else:
        # print(idx)
        current_points = self.data[idx, ...].float()
        #current_points = torch.from_numpy(current_points[:self.num_points, :]).float()
        #label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)
        # current_points = (current_points[:self.num_points, :]).float()
        label = (self.labels[idx, ...]).long()
        part_id = (self.part_id[idx, ...]).long()
        # print(current_points.shape)
        # print(current_points)
        # print(label)
        # print(part_id)
        return current_points, label, part_id
    
    def __len__(self):
        return self.data.shape[0]
    
    # def randomize(self, idx):
    #     pt_idxs = np.arange(0, self.num_points)
    #     np.random.shuffle(pt_idxs)
    #     return self.data[idx,pt_idxs].copy()
    
    # def read_classes_ShapeNet(self):
    #     BASE_DIR = os.path.dirname(os.path.abspath('hdf5_data'))
    #     DATA_DIR = os.path.join(BASE_DIR,'hdf5_data')
    #     file = open(os.path.join(DATA_DIR, 'all_object_categories_Copy.txt'), 'r')
    #     shape_names = file.read()
    #     shape_names= np.array(shape_names.split('\n')[:-1])
    #     return shape_names


# def main():
#     BASE_DIR = "../../"
#     DATA_DIR = os.path.join(BASE_DIR,'train','pts')
#     LABEL_DIR = os.path.join(BASE_DIR,'train','label')
#     pcd_size = 2907
#     all_data = []
#     all_label = []
#     all_part_id = []
#     for _, _, fs in os.walk(DATA_DIR):
#         print(len(fs))
#         for f in tqdm(fs):
#             file_name = f[:-4]
#             label_path = os.path.join(LABEL_DIR, file_name+".txt")
#             pcd_path = os.path.join(DATA_DIR, f)
#             # load pcd and label data
#             data = read_pts(pcd_path)
#             part_id = read_label(label_path)

#             # pad for alignment
#             pad_size = pcd_size - data.shape[0]
#             data = np.pad(data, ((pad_size//2, pad_size - (pad_size//2)), (0, 0)), 'edge')
#             part_id = np.pad(part_id, ((pad_size//2, pad_size - (pad_size//2))), 'edge')
#             label = np.ones(part_id.shape)

#             # update the complete dataset
#             all_data.append(data)
#             all_label.append(label)
#             all_part_id.append(part_id)
#     all_data = torch.from_numpy(np.concatenate(all_data, axis = 0))
#     all_label = torch.from_numpy(np.concatenate(all_label, axis= 0))
#     all_part_id = torch.from_numpy(np.concatenate(all_part_id, axis= 0))
#     print(all_data.shape)
#     print(all_label.shape)
#     print(all_part_id.shape)
#     # a=read_pts(f"{BASE_DIR}train/pts/1ab8a3b55c14a7b27eaeab1f0c9120b7.pts")
#     # a=read_label(f"{LABEL_DIR}/1ab8a3b55c14a7b27eaeab1f0c9120b7.txt")
#     # print(a.dtype)
#     # print(a.shape)
#     # b=np.ones(a.shape)
#     # print(b)
#     # print(b.shape)
#     # print(a)
#     # print("hh")
#     # for root, ds, fs in os.walk(LABEL_DIR):
#     #     print(fs)
#     return 0

# main()
# #D = ShapeNetData(train = True, validation = False, num_points = 2048, randomize_data = False)
# #print(D.part_id.size())

