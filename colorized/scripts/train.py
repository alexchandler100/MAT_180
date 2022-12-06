# takes many input parameters
from forward_process import q_sample
import torch.nn.functional as F
import torch

# the loss function
def p_loss(model,x0,cond_x,t,sqrt_alphas_cumuluative_prods,sqrt_one_minus_alphas_cumulative_prods):
    noise = torch.randn_like(x0)

    x_t = q_sample(x0,t,sqrt_alphas_cumuluative_prods,sqrt_one_minus_alphas_cumulative_prods)
    epsilon_theta = model(x_t,cond_x,t)

    loss = F.l1_loss(noise,epsilon_theta)
    return loss

def train():
    pass