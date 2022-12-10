from torch import nn
import torch
from einops import rearrange
import math


# core block of resnet so: convolution, groupnorm and silu activation function
class CoreBlock(nn.Module):
    # groups number for group normalization
    def __init__(self,dim_in,dim_out,num_groups=8):
        super().__init__()
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
class ResNetBlock(nn.Module):
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
        time_embedd = self.mlp_timestep(time_embedd)

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
        return out



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
        # Take the exponential of it to get rid of the log we applied and come back
        # to the original expression
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
    def __init__(self,dim,f):
        super().__init__()
        self.f = f
        self.norm = nn.GroupNorm(1,dim)

    def forward(self,x):
        x = self.norm()
        x = self.f(x)
        return x


# upsample block (can also be defined inside customUnet)
def upSample(dim):
    return nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1)

# downsample block (can also be defined inside customUnet)
def downSample(dim):
    return nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1)

# the custom unet model that uses the blocks defined above
class CustomConditionalUNet(nn.Module):
    def __init__(self,  image_channels=3,
                        conditional_image_channels=3,
                        up_channels=(256,128,64,32),
                        down_channels=(32,64,128,256),
                        time_emb_dim=32):            
        super().__init__()

        pos_emb = SinusoidalPositionEmbeddings(time_emb_dim)
        self.timestep_MLP = nn.Sequential(
                            pos_emb,
                            nn.Linear(time_emb_dim,time_emb_dim),
                            nn.GELU()
                        )
        
        # The condional image is concatenated with the noised image
        # so we add the number of channels
        input_channel = image_channels + conditional_image_channels

        # Initial convolution
        self.init_conv = nn.Conv2d(input_channel,down_channels[0],kernel_size=3,padding=1)

        # Fill downsample blocks
        self.downs = nn.ModuleList([])
        
        for i in range(len(down_channels)-1):
            self.downs.append(
                nn.ModuleList([
                    ResNetBlock(down_channels[i],down_channels[i+1],time_emb_dim),
                    ResNetBlock(down_channels[i+1],down_channels[i+1],time_emb_dim),
                    # TODO Add Residual -> prenorm -> linear attention
                    # Don't downsample if it is the last down block
                    downSample(down_channels[i+1]) if i!=(len(down_channels)-2) else nn.Identity()
                ])
            )
        
        # Bottleneck layers
        mid_dim = down_channels[-1]
        self.bottleneck_block1 = ResNetBlock(mid_dim,mid_dim,time_emb_dim)
        # TODO Add Residual -> prenorm -> attention
        self.bottleneck_block2 = ResNetBlock(mid_dim,mid_dim,time_emb_dim)

        # Fill upsample blocks
        self.ups = nn.ModuleList([])

        for i in range(len(up_channels)-1):            
            self.ups.append(
                nn.ModuleList([
                    ResNetBlock(up_channels[i]*2,up_channels[i+1],time_emb_dim),
                    ResNetBlock(up_channels[i+1],up_channels[i+1],time_emb_dim),
                    # TODO Add Residual -> prenorm -> linear attention
                    # Don't upsample if it is the last up block
                    upSample(up_channels[i+1]) if i!=(len(up_channels)-2) else nn.Identity()
                ])
            )

        # output layer
        self.lastblock = ResNetBlock(up_channels[-1],up_channels[-1],time_emb_dim)
        self.lastconv = nn.Conv2d(up_channels[-1],image_channels,1)

    def forward(self, x, cond_x, timestep):

        # Concat x and cond_x on the channel axis
        x = torch.cat((x,cond_x),dim=1)

        x = self.init_conv(x)

        t = self.timestep_MLP(timestep)

        # downsample
        skip_connections=[]

        for block1,block2,downsample in self.downs:
            x = block1(x,t)
            x = block2(x,t)
            skip_connections.append(x)
            x = downsample(x)
            
        # Bottleneck
        x = self.bottleneck_block1(x,t)
        x = self.bottleneck_block2(x,t)
        
        # upsample
        for block1,block2,upsample in self.ups:
            residual = skip_connections.pop()
            # Add the residual x as an additional channels
            x = torch.cat((x,residual),dim=1)
            x = block1(x,t)
            x = block2(x,t)
            x = upsample(x)

        # final block 
        x = self.lastblock(x,t)

        # output layer
        x = self.lastconv(x)

        return x
        


#model = CustomConditionalUNet()
#print("Num params: ", sum(p.numel() for p in model.parameters()))
#print(model)
