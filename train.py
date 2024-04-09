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
import pandas as pd

# dataloader init
n_pts = 2907
dataset = ShapeNetData(BASE_DIR="../", train = True, validation = False, num_points = n_pts, randomize_data = False)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True,num_workers=0)


# model init
num_class = 4
model =  PointNetSegmenter(k=num_class, feature_transform=False)
FILE = "model.pth"
torch.save(model.state_dict(),FILE)


part_segmenter = PointNetSegmenter(k=num_class, feature_transform=False)   #k= num_classes
part_segmenter.load_state_dict(torch.load(FILE))
part_segmenter.cuda()



optimizer = optim.Adam(part_segmenter.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


try:
    os.makedirs("Trained_Seg_Model")
except OSError:
    pass

# logging utilities
training_df= pd.DataFrame(columns=["epoch", "i", "num_batch", "train_loss", "accuracy"] )
validation_df= pd.DataFrame(columns=["epoch", "i", "num_batch", "validation_loss", "accuracy"] )
    

batchSize = 32
num_batch = len(dataset) /batchSize
num_epoch = 120

# # initialize cuDNN
# def force_cudnn_initialization():
#     s = 32
#     dev = torch.device('cuda')
#     torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
# force_cudnn_initialization()

# training
for epoch in range(num_epoch):
    scheduler.step()
    for i, data in tqdm(enumerate(dataloader, 0)):
        points,class_label,target = data
        points = points.transpose(2, 1)
        points, class_label, target = points.cuda(),class_label.cuda(),target.cuda()
        part_segmenter = part_segmenter.train()
        
        # forwardpass
        pred, trans, trans_feat = part_segmenter(points)
        pred = pred.view(-1, num_class)
        target = target.view(-1, 1)[:, 0]
        loss = F.nll_loss(pred, target)
        
        # Backwardpass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item()/float(batchSize * n_pts)))
        training_df= training_df.append({"epoch": epoch, "i":i, "num_batch":num_batch, "train_loss":loss.item(), "accuracy":correct.item()/float(batchSize * n_pts)},ignore_index=True)
    torch.save(part_segmenter.state_dict(), 'Trained_Seg_Model/seg_model_%d.pth' % (epoch))
    training_df.to_csv('training.csv')    
    validation_df.to_csv('validation.csv') 
