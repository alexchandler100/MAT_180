from torch import nn
import torch
from einops import rearrange
import math


# core block of resnet so: convolution, groupnorm and silu activation function
class CoreBlock(nn.Module):
    # groups number for group normalization
    def __init__(self,dim_in,dim_out,num_groups=8):
        super.__init__()
        self.conv = nn.Conv2d(dim_in,dim_out,3,padding=1)
        self.norm = nn.GroupNorm(num_groups,dim_out)
        self.activ = nn.SiLU() # Sigmoid Linear Unit (SiLU) function

    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activ(x)
        return x

# Take timestep embedding into an MLP
class TimestepEmbeddingMLP(nn.Module):
    def __init__(self,time_embedd_dim,dim_out):
        super().__init__()
        self.activ = nn.SiLU()
        self.dense = nn.Linear(time_embedd_dim,dim_out)

    def forward(self,x):
        x = self.activ(x)
        x = self.dense(x)
        return x
    

# Resnet block with timestep embedding
class ResNet(nn.Module):
    def __init__(self,dim_in,dim_out,time_embedd_dim,num_groups=8):
        super().__init__()

        self.block1 = CoreBlock(dim_in,dim_out,num_groups)
        self.block2 = CoreBlock(dim_out,dim_out,num_groups)

        if(dim_in!=dim_out):
            # If dim_in differs from dim_out, pass input to a conv layer 
            # that output the dim_out dimension
            self.residual_conv = nn.Conv2d(dim_in,dim_out,1)
        else:
            # If the dims are equal (a conv layer was applied before)
            # just apply identity
            self.residual_conv = nn.Identity()

        self.mlp_timestep = TimestepEmbeddingMLP(time_embedd_dim,dim_out)

    def forward(self,x,time_embedd):

        # x goes through first block
        h1 = self.block1(x)

        # pass the embedded time step into an MLP that output a tensor of shape dim_out
        # which is compatible with h1
        time_embedd = self.mlp(time_embedd)

        # Rearrange the shape of the tensor time_embedd which has shape (b,c) 
        # to (b,c,1,1) so we can add the output of h1 to it
        # b = batch_size, c=channel
        h2 = rearrange(time_embedd,"b c -> b c 1 1")
        
        # Add the output of the MLP with the output of the first block
        h2 = h2 + h1 

        # give this output through the second block
        h2 = self.block2(h2)
        
        # add the skip connection (residual) to the output
        out = h2 + self.residual_conv(x)



# sinusoidal positional embeddings that allow to embed the timestep t
# Implemented the same way asthe original paper DDPM 
# https://github.com/lucidrains/denoising-diffusion-pytorch
# It takes a tensor of shape (batch_size, 1) as input and turns it into a tensor
# of shape (batch_size,dim)
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self,dim):
        super().__init__()
        # output dimension of the embedding
        self.dim = dim

    def forward(self,timestep):
        # To send computation on the right device
        device = timestep.device
        # Take half the size of the required dimension
        half_dim = self.dim // 2
        # Take the log of 10 000 and scale it by half the dimension required -1
        embeddings = math.log(10000) / (half_dim -1)
        # Create a 1D Tensor containing half_dim values going from 0 to half_dim -1
        arange = torch.arange(half_dim,device=device)
        # Multiply the embeddings negative value element-wise to this tensor
        embeddings = arange * -embeddings
        # Take the exponential of it
        embeddings = torch.exp(embeddings)

        # Add an empty last dimension to the timestep tensor so it has 
        # shapes (batch_size,1,1)
        t_augmented = timestep[:,None]
        # Add an empty first dimension to the embeddings tensor so it 
        # has shapes (1,half_dim)
        emb_augmented = embeddings[None,:]

        # Multpily the two augmented tensors so we obtain a tensor of shape   
        # (batch_size,1,half_dim)
        embeddings = t_augmented * emb_augmented

        # Take the sin element-wise to the values of the tensor
        emb_sin = embeddings.sin()
        # Take the cos element-wise to the values of the tensor
        emb_cos = embeddings.cos()

        # Concatenate the last dimension of these tensors, so in the end we get
        # a tensor of shape (batch_size,1,self.dim)
        embeddings = torch.cat((emb_sin,emb_cos),dim=-1)
        
        return embeddings

    

# Optional for now
class AttentionHeads(nn.Module):
    pass

class LinearAttentionHeads(nn.Module):
    pass

# Should be applied before Attention layers if implemented:
class PreGroupNorm(nn.Module):
    pass

# upsample block (can also be defined inside customUnet)
def upSample(dim):
    return nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1)

# downsample block (can also be defined inside customUnet)
def downSample(dim):
    return nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1)

# the custom unet model that uses the blocks defined above
class CustomUNet(nn.Module):
    pass

# the loss function
def p_loss(model,x0,t):
    pass