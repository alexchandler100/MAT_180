import torch
import sys
sys.path.append('../')
from scripts import util
import gc



# calls the model to predict the noise in the image and return the denoised image
# it is the algorithm 2 in ddpm paper
@torch.no_grad()
def p_sample(model, x, cond_x, t, t_index, constantDiffusionTerms):
    # match batch timesteps to various corresponding values 
    sqrt_alphas_inverse_t = util.extract_t_th_value_of_list(constantDiffusionTerms.sqrt_alphas_inverse, t, x.shape)
    sqrt_one_minus_alphas_cumulative_prods_t = util.extract_t_th_value_of_list(constantDiffusionTerms.sqrt_one_minus_alphas_cumulative_prods, t, x.shape)
    betas_t = util.extract_t_th_value_of_list(constantDiffusionTerms.betas, t, x.shape)
    
    # calculate the (noise) parameterized mean using our UNet
    parameterized_model_mean = sqrt_alphas_inverse_t * (x - betas_t * model(x, cond_x, t) / sqrt_one_minus_alphas_cumulative_prods_t)
    # If we have completed the entire reverse process, return the parameterized mean
    # else generate random noise (epsilon), scale by sqrt(beta) and add to model mean 
    # (reparametrization trick)
    if (t_index == 0):
        return parameterized_model_mean
    else:
        noise = torch.randn_like(x)
        sqrt_betas_tilde_t = util.extract_t_th_value_of_list(constantDiffusionTerms.sqrt_betas_tilde,t,x.shape)
        return parameterized_model_mean + (sqrt_betas_tilde_t * noise)
    

# same but start from pure noise x_T and loop until getting a new x_0
@torch.no_grad()
def p_sample_loop(model, shape, device, timesteps, cond_x, constantDiffusionTerms):
    #shape=(batch_size, channels, image_width, image_height)
    batch_size = shape[0]
    
    # sample pure noise in the correct output shape (once for each in batch)
    noisy_image = torch.randn(shape, device=device)
    noisy_images = []

    # for each timestep until 0, remove some noise
    for t in reversed(range(0, timesteps)):
        print(t)

        # copy the timestep t into a vector of size batch_size
        batch_sized_timestep = torch.full((batch_size,), t, device=device) 
        # sample noisi image at timestep t
        noisy_image = p_sample(model, 
                                noisy_image, cond_x,
                                batch_sized_timestep, t,
                                constantDiffusionTerms)
        if(t%30==0):
            print("saving!")
            noisy_images.append(noisy_image.detach().cpu().numpy())
    
    # return all noisy images at every timestep
    return  noisy_images