import torch
from util import extract_t_th_value_of_list 
# calls the model to predict the noise in the image and return the denoised image
# it is the algorithm 2 in ddpm paper
def p_sample(model, x, cond_x, t, t_index, sqrt_alphas_inverse, sqrt_one_minus_alphas_cumulative_prods, betas, sqrt_betas):
    # match batch timesteps to various corresponding values 
    sqrt_alphas_inverse = extract_t_th_value_of_list(sqrt_alphas_inverse, t, x.shape)
    sqrt_one_minus_alphas_cumulative_prods = extract_t_th_value_of_list(sqrt_one_minus_alphas_cumulative_prods, t, x.shape)
    betas_t = extract_t_th_value_of_list(betas, t, x.shape)
    
    # calculate the (noise) parameterized mean using our UNet
    parameterized_model_mean = sqrt_alphas_inverse * (x - betas_t * model(x, cond_x, t) / sqrt_one_minus_alphas_cumulative_prods)

    # If we have completed the entire reverse process, return the parameterized mean
    # else generate random noise (epsilon), scale by sqrt(beta) and add to model mean
    if (t_index == 0):
        return parameterized_model_mean
    else:
        noise = torch.randn_like(x)
        sqrt_betas = torch.sqrt(betas)
        return parameterized_model_mean + (sqrt_betas * noise)
    

# same but start from pure noise x_T and loop until getting a new x_0
def p_sample_loop(model,x):
    pass

