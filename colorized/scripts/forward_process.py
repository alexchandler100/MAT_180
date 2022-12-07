import torch
from util import extract_t_th_value_of_list, ConstantDiffusionTerms

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

# takes x0 as input, a timestep t and output x_t
def q_sample(x0,t,constantDiffusionTerms:ConstantDiffusionTerms):
    # gaussian noise of size x0 (the first image)
    epsilon = torch.rand_like(x0)

    # extract the t^th index of sqrt(alphas_cumulative_prods) for each batch
    sqrt_alphas_cumuluative_prods_t = extract_t_th_value_of_list(constantDiffusionTerms.sqrt_alphas_cumuluative_prods,t,x0.shape)

    # extract the t^th index of sqrt(1 - alphas cumulative products) for each batch
    sqrt_one_minus_alphas_cumulative_prods_t = extract_t_th_value_of_list(constantDiffusionTerms.sqrt_one_minus_alphas_cumulative_prods,t,x0.shape)

    # formula obtained using the reparametrization trick
    q_sample_t = sqrt_alphas_cumuluative_prods_t * x0 + sqrt_one_minus_alphas_cumulative_prods_t * epsilon

    return q_sample_t