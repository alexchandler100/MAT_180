#!/usr/bin/env python
# coding: utf-8

# # Reverse Process

# ## Imports

# In[1]:


import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
# from cv2 import cvtColor, COLOR_BGR2RGB, COLOR_LAB2RGB, COLOR_LAB2BGR
import sys
from einops import rearrange

sys.path.append('../')
#from util import LABtoRGB
from scripts import customUNet as unet
from scripts import forward_process as fp, reverse_process as rp, train, util


# ## Load image from dataset

# In[2]:


ab1 = np.load("../data/image-colorization/ab/ab/ab1.npy")
ab2 = np.load("../data/image-colorization/ab/ab/ab2.npy")
ab3 = np.load("../data/image-colorization/ab/ab/ab3.npy")
l = np.load("../data/image-colorization/l/gray_scale.npy")
# Concatenate the numpy files
ab = np.concatenate((ab1,ab2,ab3),axis=0)


# In[3]:


lab = np.zeros((1,224,224,3))
img_index= 5000 # not used during training
lab[:,:,:,0] = l[img_index,:]
lab[:,:,:,1:] = ab[img_index,:]    

lab = lab[0].astype("uint8")
lab.shape


# In[4]:


size=64
img = cv2.normalize(lab, None, alpha = -1, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
img = cv2.resize(lab, dsize=(size, size), interpolation=cv2.INTER_CUBIC)

img = torch.tensor(np.array(img))
# Reshape so the tensor has the right shape for training
img = rearrange(img,"h w c-> 1 c h w")
img.shape


# In[5]:


# Extract the grayscale (l) color channel
gray = l[img_index,:]
gray = torch.tensor(gray)
gray = rearrange(gray,"w h-> 1 h w 1")
# Tripple the last dimension so it is the same shape as the color image
gray = gray.expand(gray.shape[0],gray.shape[1],gray.shape[2],3)
gray = rearrange(gray,"b h w c-> b c h w")
gray.shape


# ## Load model

# In[6]:


T = 10
diffTerms = util.ConstantDiffusionTerms(T, fp.linear_beta_schedule)
model = unet.CustomConditionalUNet()
model.load_state_dict(torch.load("../models/modelv1.pth"))
device = util.set_device()
model.to(device);


# ## Sample

# In[ ]:


noisy_images = rp.p_sample_loop(model,gray.shape,device,T,gray,diffTerms)

