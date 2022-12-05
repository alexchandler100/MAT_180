import torch
import math

# Define cosine beta schedule
# As proposed by https://arxiv.org/pdf/2102.09672.pdf
# s is an offset s to prevent Beta_t from being too small near t = 0
def cosine_beta_schedule(timesteps,s=0.008):
    steps = timesteps+1
    # Create a vector starting from 0 to timesteps in `steps` steps
    x = torch.linspace(0,timesteps,steps)
    # Normalize it by dividing by timesteps
    x = x/timesteps

    # Use the formula presented in the IDDPM paper
    alphas_cumulative_prod = torch.cos((x + s)/(1+s) * (torch.pi * 0.5))**2
    alphas_cumulative_prod = alphas_cumulative_prod / alphas_cumulative_prod[0]
    
    # go from alpha to beta with the formula defined in the paper
    betas = 1 - alphas_cumulative_prod[1:] / alphas_cumulative_prod[:-1]

    # Clip Beta_t to be no larger than 0.9999 near t=T and not too close to 0 near t=0
    betas = torch.clip(betas, 0.0001, 0.9999)

    return betas


def linear_beta_schedule(timesteps,start=0.0001,end=0.02):
    # Take the same values as the DDPM paper of 2020
    return torch.linspace(start,end,timesteps)

# Define forward_diffusion_sample or q_sample or another coherent name:
# takes x0 as input, a timestep t and output x_t
def q_sample(x0,t):
    pass

