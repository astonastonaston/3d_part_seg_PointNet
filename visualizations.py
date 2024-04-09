#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import print_function
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataloader import ShapeNetData
from segmentation_model import STN3d,STNkd,PointNetfeature,PointNetSegmenter,feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import open3d as o3d



# In[4]:

n_pts = 2907
num_class = 4
FILE = "Trained_Seg_Model/seg_model_110.pth"
part_segmenter = PointNetSegmenter(k=num_class, feature_transform=False)   #k= num_classes
part_segmenter.load_state_dict(torch.load(FILE))
part_segmenter.cuda()


# data loading
def read_pts(filename):
    return np.loadtxt(filename)

pts = []
for j in range(6):
    pcd = read_pts(f"../../test/{j}.pts")
    # pad for alignment
    pad_size = n_pts - pcd.shape[0]
    pcd = np.pad(pcd, ((pad_size//2, pad_size - (pad_size//2)), (0, 0)), 'edge')
    pts.append(pcd[None, ...])
pts = torch.from_numpy(np.concatenate(pts, axis = 0))
pts = pts.float()
pts = pts.transpose(2, 1)
pts = pts.cuda()
print(pts)
print(pts.shape)
part_segmenter = part_segmenter.eval()

# prediction
pred, trans, trans_feat = part_segmenter(pts)
pred_choice = torch.max(pred, dim=2)[1]
preds = pred_choice.cpu().numpy()
pts = pts.cpu().transpose(2, 1)
print(preds)
print(preds.shape)
print(pts.shape)


# visualizations:
rgbMap = np.array([[255,0,0], [0,255,0], [0,0,255], [192,47,255]]) # red, green, blue, purple
for j in range(6):
    pcd = o3d.geometry.PointCloud()
    ptsj, ptsLabj = pts[j, ...], preds[j, ...]
    colorsj = rgbMap[ptsLabj]
    pcd.points = o3d.utility.Vector3dVector(ptsj) # XYZ points
    pcd.colors = o3d.utility.Vector3dVector(colorsj / 255.0)  #open3d requires colors (RGB) to be in range[0,1]
    o3d.visualization.draw_geometries([pcd])
