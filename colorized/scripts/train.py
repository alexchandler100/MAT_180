# takes many input parameters
from forward_process import q_sample
from util import ConstantDiffusionTerms
import torch.nn.functional as F
import torch

# the loss function
def p_loss(model,x0,cond_x,t,constantDiffusionTerms:ConstantDiffusionTerms):
    noise = torch.randn_like(x0)

    x_t = q_sample(x0,t,constantDiffusionTerms.sqrt_alphas_cumuluative_prods,
                        constantDiffusionTerms.sqrt_one_minus_alphas_cumulative_prods)
    epsilon_theta = model(x_t,cond_x,t)

    loss = F.l1_loss(noise,epsilon_theta)
    return loss

def train(epochs,device,optimizer,train_loader,batch_size,timesteps,
                constantDiffusionTerms:ConstantDiffusionTerms,model):
    
    costs = []
    for epoch in range(epochs):
        for step,(batch_imgs,batch_cond_imgs) in enumerate(train_loader):
            
            # Reset the gradients to 0
            optimizer.zero_grad()

            # Sample a timestep t from a uniform distribution for every example in the 
            # batch (long tensor)
            t = torch.randint(0,timesteps,(batch_size,),device=device).long()

            loss = p_loss(model,batch_imgs,batch_cond_imgs,t,constantDiffusionTerms)
            
            if(step%100==0):
                costs.append(loss)
                print(f"Step {step} of epoch {epoch}: Loss={loss}")
            
            loss.backward()
            optimizer.step()

            #TODO: plt show sample imaages every X epochs