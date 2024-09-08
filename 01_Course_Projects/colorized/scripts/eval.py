from ignite.metrics import FID
import torch
import torchvision

def eval(model,train_loader,test_loader,img_size):
    # https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/
    pass