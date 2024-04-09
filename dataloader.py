
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import glob
import csv
from tqdm import tqdm

def read_pts(filename):
    return np.loadtxt(filename)

def read_label(filename):
    return np.loadtxt(filename)

def load_data(train, validation, pcd_size=2907, BASE_DIR="./"):
    if ((train== True) and (validation == False)):
        partition = 'train'
    elif((train== True) and (validation == True)):
        partition = 'val'
    else:
        partition = 'test'
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


class ShapeNetData(Dataset):
    def __init__(self, train = True, validation = False, num_points = 2907, randomize_data = False, BASE_DIR = "./"):
        super(ShapeNetData, self).__init__()
        self.data, self.labels, self.part_id = load_data(train, validation, pcd_size=num_points, BASE_DIR=BASE_DIR)
        self.num_points = num_points
        self.randomize_data = randomize_data
            
    def __getitem__(self, idx):
        current_points = self.data[idx, ...].float()
        label = (self.labels[idx, ...]).long()
        label = label - 1
        part_id = (self.part_id[idx, ...]).long()
        part_id = part_id - 1
        return current_points, label, part_id
    
    def __len__(self):
        return self.data.shape[0]
    
