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
def p_sample_loop(model, shape, device, timesteps, cond_x, sqrt_alphas_inverse, sqrt_one_minus_alphas_cumulative_prods, betas, sqrt_betas): # device parameter for now QUESTION
    batch_size = shape[0]
    
    # sample pure noise in the correct output shape (once for each in batch)
    noisy_image = torch.randn(shape, device=device)
    noisy_images = []

    # for each timestep until 0, remove some noise
    for i in reversed(range(0, timesteps)): # talk about the tdqm thing
        batch_sized_timestep = torch.full((batch_size,), i, device=device) # note that I did not specify the data type, hope that's fine
        noisy_image = p_sample(model, 
                                noisy_image, cond_x,
                                batch_sized_timestep, i,
                                sqrt_alphas_inverse, 
                                sqrt_one_minus_alphas_cumulative_prods, 
                                betas, sqrt_betas)
        noisy_images.append(noisy_image.cpu().numpy())
    
    # return all noisy images at every timestep
    return  noisy_images

def sample(model, image_width, image_height, batch_size, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_width, image_height))
    # Q: Do we need to be saving each of the images? Maybe to show progress?

