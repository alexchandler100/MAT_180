from torch import nn

# core block of resnet so: convolution, groupnorm and silu activation function
class CoreBlock(nn.Module):
    pass

# Resnet block
class ResNet(nn.Module):
    pass

# sinusoidal positional embeddings that allow to embed the timestep t
class PositionEmbeddings(nn.Module):
    pass

# Optional for now
class AttentionHeads(nn.Module):
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