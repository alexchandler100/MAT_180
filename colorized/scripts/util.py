import torch
import torch.nn.functional as F

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


# Contains all the constant terms that we use often during sampling
class ConstantDiffusionTerms:
    # schedule method can be cosine or linear in our implementation
    def __init__(self,T,schedule_method):

        self.betas = schedule_method(T)

        # Compute the alphas
        self.alphas = 1 - self.betas

        # Compute the cumulative product of alphas (alpha^bar_T)
        self.alphas_cumulative_prods = torch.cumprod(self.alphas,axis=0)

        # Compute alpha^bar_{T-1}
        # Remove the last value and pad the tensor only at the beginning by adding a 1
        # so it has the same dimension of alphas_cumulative_prod but exclude the alpha_t in the product
        self.alphas_cumulative_prods_minus_1 = F.pad(self.alphas_cumulative_prods[:-1],(1,0),value=1.0)

        # Compute 1/sqrt(alphas) or sqrt(1/alphas)
        self.sqrt_alphas_inverse = torch.sqrt(1/self.alphas)

        # Calculations used for the forward diffusion process: q(x_t | x_{t-1})
        self.sqrt_alphas_cumuluative_prods = torch.sqrt(self.alphas_cumulative_prods)
        self.sqrt_one_minus_alphas_cumulative_prods = torch.sqrt(1 - self.alphas_cumulative_prods)

        # Compute beta^tilde_t (which is not learned here, so we have a closed form)
        # see the theory file for more information
        self.betas_tilde = self.betas * (1 - self.alphas_cumulative_prods_minus_1) / (1 - self.alphas_cumulative_prods)

        self.sqrt_betas_tilde = torch.sqrt(self.betas_tilde)


# Extract the appropriate t indexes in a list of values, for a batch of indices 
def extract_t_th_value_of_list(values,t,x_shape):
    batch_size = t.shape[0]
    # Extract the elements at indices t (there is batch_size t values) in the list of values
    res = values.gather(-1,t.cpu())

    # output the values in the same shape as the image x
    additional_dims = (1,) * (len(x_shape)-1)
    # Reshape the result with the empty dimensions to match the image
    res = res.reshape(batch_size,*additional_dims).to(t.device())
    return res

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