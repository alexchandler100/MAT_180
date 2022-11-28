<!-- #region -->
# COLORIZED - Colorization Of Lost Or Ruined ImageZ Employing Diffusion

Clément Weinreich

Brennan Whitfield

Vivek Shome

## Abstract
  
### How will the data be collected?

The data will be collected from a public dataset on Kaggle named [Image Colorization]([https://www.kaggle.com/datasets/shravankumar9892/image-colorization/code](https://www.kaggle.com/datasets/shravankumar9892/image-colorization/code)). This dataset consists of 50 thousand grayscale and colored images rescaled to 224 x 224.  

### What task do you want to accomplish with the data?

We seek to recolor grayscale photographs from the University Archives Photographs at Shields Library. This project could provide the library with a means of viewing a local landmark/area from the perspective of an individual present at the original time.

### What kind of learning algorithm do you propose using to accomplish this task?

We plan to accomplish this task using an image-to-image conditional diffusion model of the form $P(y | x)$, where $x$ is a grayscale image and $y$ is a color image.  

### How will you measure your performance of the task?

To measure the performance of the task, we will use the [Fréchet Inception Distance]([https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance)) (FID). The FID metric compares the distribution of generated images with the distribution of a set of real images (ground truth). The FID metric was introduced in 2017 and is the current standard metric for assessing the quality of generative models.

## Instructions (delete this once finished)
Your README.md file (located at the root of your project directory) should completely describe the project, what it accomplishes, how well it performs, and walk through how a new user can use your model on their own data after cloning the repository. This means, by following instructions in your README.md file, I should easily be able to run your algorithm on a new example not in your dataset and see that the model accomplishes what it promised to accomplish (at least up to your claimed performance measure).

## Introduction

outline this file? Merge with abstract?

## Theory

### General idea


The idea of diffusion models is to slowly destroy structure in a data distribution through an iterative forward process. Then, we learn a reverse diffusion process using an neural network, that restores structure in data. This model yield to a highly flexible generative model of the data. This can be seen as a Markov chain of diffusion steps, which slowly add random noise to the data, and then learn to reverse the diffusion process in order to construct new desired data samples from the noise.

In the case of image colorization, we use a conditional diffusion model which takes an additional input (a grayscale image) in order to restore the color of this image. 

### Notation

The following notation will be adopted for the next parts:

- $\mathcal{N}(x;\mu,\sigma^2)$ : sampling x from a normal distribution of mean $\mu$ and variance $\sigma^2$
- $\mathbf{x_t}$ is the image after applying $t$ iterations of noise through the forward process
- $\mathbf{x_0}$ is the original image
- $\mathbf{x_T}$ is the final image of the forward process which follows an isotropic gaussian distribution ($T$ is constant)
- $q(\mathbf{x_t}|\mathbf{x_{t-1}})$ corresponds to the forward process, taking an image $\mathbf{x_{t-1}}$ as input, and output $\mathbf{x_t}$ which contains more noise
- $p_\theta(\mathbf{x_{t-1}}|\mathbf{x_t})$ corresponds to the reverse process, taking an image $\mathbf{x_t}$ as input, and output $\mathbf{x_{t-1}}$ which contains less noise

### The forward process

Let's sample an image from a real data distribution $\mathbf{x_0} \sim q(\mathbf{x})$. We define a forward diffusion process, in which a small amount of gaussian noise is iteratively added to the image $\mathbf{x_0}$, in $T$ steps, leading to the sequence of noisy images $\mathbf{x_1},\dots,\mathbf{x_T}$. The step size is controlled by a variance schedule $\beta_t$ going from 0 to 1, in $T$ steps, starting at $t=1$. The noise added is sampled from a gaussian distribution. Thus we can define:
$$q(\mathbf{x_t}|\mathbf{x_{t-1}}) = \mathcal{N}(\mathbf{x_t};\sqrt{1-\beta_t}\mathbf{x_{t-1}},\beta_t\mathbf{I})$$
Where the variance schedule scales the mean and the variance of the noise sampled from the normal distribution. Since our forward process is a Markov Chain (satisfying Markov property), we can also write:

$$\begin{align}
q(\mathbf{x_{1:T}}|\mathbf{x_0}) &= q(\mathbf{x_1}, \dots, \mathbf{x_T} | \mathbf{x_0}) \\
               &= \frac{q(\mathbf{x_0}, \mathbf{x_1}, \dots, \mathbf{x_T})}{q(\mathbf{x_0})} &&\text{(Bayes' Theorem)}\\
               &= \frac{q(\mathbf{x_0})q(\mathbf{x_1}|\mathbf{x_0})\dots q(\mathbf{x_T}|\mathbf{x_{T-1}})}{q(\mathbf{x_0})} &&\text{(Markov property)}\\
               &= q(x_1|x_0)\dots q(x_T|x_{T-1})\\
               &= \prod^T_{t=1}q(x_t|x_{t-1})
\end{align}$$

<!-- Reference the reparameterization trick from lilianweng? -->
Additionally, we can improve the forward process further by using a reparameterization trick, allowing us to sample a noisy image $x_t$ at any particular time $t$. **(What are the implications of being able to sample noisy images at random timesteps?)** First, we let $\alpha_{t} = 1 - \beta_t$, and we also define $\bar{\alpha_t} = \prod\nolimits_{i=1}^t \alpha_i$. Now, we rewrite 

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t;\sqrt{\alpha_t}x_{t-1},(1 - \alpha_t)\mathbf{I})$$

<!-- Should we use element wise product here for matrices? -->
Using the reparameterization trick for Gaussian distribution $\mathcal{N}(x; \mu, \sigma^2)$, $x = \mu + \sigma\epsilon $, where $\epsilon \sim \mathcal{N}(\boldsymbol{0}, \mathbf{I})$
### The reverse process

## Implementation

How we implemented it

## Results

what it accomplishes, how well it performs

## How to use our model

Walk through how a new user can use your model on their own data after cloning the repository

suggestion: create a notebook where the user can give an image, and the code to preprocess the image, run the model and everything is already included in it?

## Conclusion

feedback, improvements,...

## References

<!-- #endregion -->
