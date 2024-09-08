import numpy as np
import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
from cv2 import cvtColor, COLOR_BGR2RGB, COLOR_LAB2RGB, COLOR_LAB2BGR
import sys
from einops import rearrange

sys.path.append('../')
from scripts import customUNet as unet
from scripts import forward_process as fp, reverse_process as rp, train, util


if len(sys.argv) != 2:
    print("Incorrect Usage. Use python3 colorize.py <grayscale_image>")
    sys.exit()

#Open image
grayscale_img = cv2.imread(sys.argv[1], 0)

# Normalize & Resize
size=64
grayscale_img = cv2.normalize(grayscale_img, None, alpha = -1, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
grayscale_img = cv2.resize(grayscale_img, dsize=(size, size), interpolation=cv2.INTER_CUBIC)

# Convert to a tensor
img = torch.tensor(np.array(grayscale_img))
# Reshape so the tensor has the right shape
gray = rearrange(img,"h w -> 1 h w")

# Duplicate channels
gray = rearrange(gray,"b w h-> b w h 1")
gray = gray.expand(gray.shape[0],gray.shape[1],gray.shape[2],3)
gray = rearrange(gray,"b w h c-> b c w h")

# Load model
T = 300
diffTerms = util.ConstantDiffusionTerms(T, fp.linear_beta_schedule)
model = unet.CustomConditionalUNet()
model.load_state_dict(torch.load("../models/modelv1.pth"))
device = util.set_device()
model.to(device)

# Sample
noisy_images = rp.p_sample_loop(model,gray.shape,device,T,gray,diffTerms)

results = []
for img in noisy_images:
    img = rearrange(img,"b c w h-> b w h c")[0]
    norm = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm = norm.astype("uint8")
    img_rgb = cv2.cvtColor(norm,cv2.COLOR_LAB2RGB)
    results.append(img_rgb)

if not isinstance(results[0], list):
    results = [results]

plt.axis('off')
plt.imshow(np.asarray(results[0][-1]))
plt.show()
