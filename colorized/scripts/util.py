### NOT SURE IT WILL BE USEFUL YET
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid
import PIL
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

# Transform a PIL (base format of an image loaded with PIL) image to a pytorch tensor:
# using torchvision
# return the tensor
def transform_to_tensor(image):
    pass

# Transform a pytorch tensor to a displayable PIL image (using torchvision)
# can output the image or just plot it
def transform_to_pil(tensor):
    pass

## Convert image from LAB to RGB
def LABtoRGB(L, ab):
    L = (L + 1.) * 50
    ab = ab * 128
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb = []
    for img_Lab in Lab:
        img_rgb = lab2rgb(img_Lab)
        rgb.append(img_rgb)
    return np.stack(rgb, axis=0)

# Choose CPU/GPU for training
def set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device