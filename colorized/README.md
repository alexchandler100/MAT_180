<!-- #region -->

# COLORIZED - Colorization Of Lost Or Ruined ImageZ Employing Diffusion

Clément Weinreich

Brennan Whitfield

Vivek Shome

## Abstract

### How will the data be collected?

The data will be collected from a public dataset on Kaggle named [Image Colorization](<[https://www.kaggle.com/datasets/shravankumar9892/image-colorization/code](https://www.kaggle.com/datasets/shravankumar9892/image-colorization/code)>). This dataset consists of 50 thousand grayscale and colored images rescaled to 224 x 224.

### What task do you want to accomplish with the data?

We seek to recolor grayscale photographs from the University Archives Photographs at Shields Library. This project could provide the library with a means of viewing a local landmark/area from the perspective of an individual present at the original time.

### What kind of learning algorithm do you propose using to accomplish this task?

We plan to accomplish this task using an image-to-image conditional diffusion model of the form $P(y | x)$, where $x$ is a grayscale image and $y$ is a color image.

### How will you measure your performance of the task?

To measure the performance of the task, we will use the [Fréchet Inception Distance](<[https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance)>) (FID). The FID metric compares the distribution of generated images with the distribution of a set of real images (ground truth). The FID metric was introduced in 2017 and is the current standard metric for assessing the quality of generative models.

## Instructions (delete this once finished)

Your README.md file (located at the root of your project directory) should completely describe the project, what it accomplishes, how well it performs, and walk through how a new user can use your model on their own data after cloning the repository. This means, by following instructions in your README.md file, I should easily be able to run your algorithm on a new example not in your dataset and see that the model accomplishes what it promised to accomplish (at least up to your claimed performance measure).

## Introduction

outline this file? Merge with abstract?

## Theory

### General Idea

The idea of diffusion models is to slowly destroy structure in a data distribution through an iterative forward process. Then, we learn a reverse diffusion process using an neural network, that restores structure in data. This model yield to a highly flexible generative model of the data. This can be seen as a Markov chain of diffusion steps, which slowly add random noise to the data, and then learn to reverse the diffusion process in order to construct new desired data samples from the noise.

In the case of image colorization, we use a conditional diffusion model which, on top of requiring a noisy image and timestep, takes a grayscale image with the intent of recoloring said image.

### Notation

The following notation will be adopted for the next parts:

-   $\mathcal{N}(x;\mu,\sigma^2)$ : sampling x from a normal distribution of mean $\mu$ and variance $\sigma^2$
-   $\mathbf{x_t}$ is the image after applying $t$ iterations of noise through the forward process
-   $\mathbf{x_0}$ is the original image
-   $\mathbf{x_T}$ is the final image of the forward process which follows an isotropic gaussian distribution ($T$ is constant)
-   $q(\mathbf{x_t}|\mathbf{x_{t-1}})$ corresponds to the forward process, taking an image $\mathbf{x_{t-1}}$ as input, and output $\mathbf{x_t}$ which contains more noise
-   $p_\theta(\mathbf{x_{t-1}}|\mathbf{x_t})$ corresponds to the reverse process, taking an image $\mathbf{x_t}$ as input, and output $\mathbf{x_{t-1}}$ which contains less noise

### The Forward Diffusion Process

Let's sample an image from a real data distribution $\mathbf{x_0} \sim q(\mathbf{x})$. We define a forward diffusion process, in which a small amount of gaussian noise is iteratively added to the image $\mathbf{x_0}$, in $T$ steps, leading to the sequence of noisy images $\mathbf{x_1},\dots,\mathbf{x_T}$. The step size is controlled by a variance schedule $\beta_t$ going from 0 to 1, in $T$ steps, starting at $t=1$. The noise added is sampled from a gaussian distribution. Thus we can define:
$$q(\mathbf{x_t}|\mathbf{x_{t-1}}) = \mathcal{N}(\mathbf{x_t};\sqrt{1-\beta_t}\mathbf{x_{t-1}},\beta_t\mathbf{I})$$
Where the variance schedule scales the mean and the variance of the noise sampled from the normal distribution. Since our forward process is a Markov Chain (satisfying Markov property), we can also write:

$$
\begin{align}
q(\mathbf{x_{1:T}}|\mathbf{x_0}) &= q(\mathbf{x_1}, \dots, \mathbf{x_T} | \mathbf{x_0}) \\
               &= \frac{q(\mathbf{x_0}, \mathbf{x_1}, \dots, \mathbf{x_T})}{q(\mathbf{x_0})} &&\text{(Bayes' Theorem)}\\
               &= \frac{q(\mathbf{x_0})q(\mathbf{x_1}|\mathbf{x_0})\dots q(\mathbf{x_T}|\mathbf{x_{T-1}})}{q(\mathbf{x_0})} &&\text{(Markov property)}\\
               &= q(\mathbf{x_1}|\mathbf{x_0})\dots q(\mathbf{x_T}|\mathbf{x_{T-1}})\\
               &= \prod^T_{t=1}q(\mathbf{x_t}|\mathbf{x_{t-1}})
\end{align}
$$

<!-- Reference the reparameterization trick from lilianweng? -->
<!-- mention how using a predictable noise schedule mitigates this-->

Additionally, we can improve the forward process further, allowing us to sample a noisy image $\mathbf{x_t}$ at any particular time $t$. First, we let $\alpha_{t} = 1 - \beta_t$, and we also define $\bar{\alpha_t} = \prod\nolimits_{i=1}^t \alpha_i$. Now, we rewrite:

$$q(\mathbf{x_t}|\mathbf{x_{t-1}}) = \mathcal{N}(\mathbf{x_t};\sqrt{\alpha_t}\mathbf{x_{t-1}},(1 - \alpha_t)\mathbf{I})$$

<!-- Should we use element wise product here for matrices? -->

using the reparameterization trick for Gaussian distribution $\mathcal{N}(\mathbf{x}; \mathbf{\mu}, \sigma^2\mathbf{I})$, $\mathbf{x} = \mathbf{\mu} + \sigma\mathbf{\epsilon} $, where $\mathbf{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \mathbf{I})$. This gives us:

$$
\begin{align}
\mathbf{x_t} &= \sqrt{\alpha_t} \mathbf{x_{t-1}} + \sqrt{1-\alpha_t}\mathbf{\epsilon_{t-1}} \\
\mathbf{x_{t-1}} &= \sqrt{\alpha_{t-1}} \mathbf{x_{t-2}} + \sqrt{1-\alpha_{t-1}}\mathbf{\epsilon_{t-2}} && \text{(Repeat for $\mathbf{x_{t-1})}$)}
\end{align}
$$

Now, we combine the above equations:

$$
\begin{align}
\mathbf{x_t} &= \sqrt{\alpha_t} \mathbf{x_{t-1}} + \sqrt{1-\alpha_t}\mathbf{\epsilon_{t-1}} \\
             &= \sqrt{\alpha_t} (\sqrt{\alpha_{t-1}} \mathbf{x_{t-2}} + \sqrt{1-\alpha_{t-1}}\mathbf{\epsilon_{t-2}}) + \sqrt{1-\alpha_t}\mathbf{\epsilon_{t-1}} \\
             &= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x_{t-2}} + \sqrt{\alpha_t(1-\alpha_{t-1})}\mathbf{\epsilon_{t-2}} + \sqrt{1-\alpha_t}\mathbf{\epsilon_{t-1}}
\end{align}
$$

Note that we can reverse the reparameterization trick separately on $\sqrt{\alpha_t(1-\alpha_{t-1})}\mathbf{\epsilon_{t-2}}$ and $\sqrt{1-\alpha_t}\mathbf{\epsilon_{t-1}}$, considering them to be samples from $\mathcal{N}(\mathbf{0},\alpha_t(1-\alpha_{t-1})\mathbf{I})$ and from $\mathcal{N}(\mathbf{0},(1-\alpha_t)\mathbf{I})$ respectively. Thus, we can combine these Gaussian distributions with different variance as $\mathcal{N}(\mathbf{0}, ((1-\alpha_t) + \alpha_t(1-\alpha_{t-1}))\mathbf{I}) = \mathcal{N}(\mathbf{0}, (1-\alpha_{t} \alpha_{t-1})\mathbf{I})$. Once again reparameterizing this new distribution:

$$
\begin{align}
\mathbf{x_t} &= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x_{t-2}} + \sqrt{1-\alpha_{t} \alpha_{t-1}}\mathbf{\epsilon} && \mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})
\end{align}
$$

This process can be continued all the way to $\mathbf{x_0}$ resulting in:

$$
\begin{align}
\mathbf{x_t} &= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x_{t-2}} + \sqrt{1-\alpha_{t} \alpha_{t-1}}\mathbf{\epsilon} \\
             &\vdots \\
\mathbf{x_t} &= \sqrt{\alpha_t \alpha_{t-1} \dots \alpha_{1}}\mathbf{x_0} + \sqrt{1-\alpha_t \alpha_{t-1} \dots \alpha_{1}}\mathbf{\epsilon} \\
             &= \sqrt{\bar{\alpha_t}}\mathbf{x_0} + \sqrt{1 - \bar{\alpha_t}}\mathbf{\epsilon}
\end{align}
$$

Since $x_t = \sqrt{\bar{\alpha_t}}\mathbf{x_0} + \sqrt{1 - \bar{\alpha_t}}\mathbf{\epsilon}$, we can once more reverse the reparamaterization process to achieve:

$$q(\mathbf{x_t} | \mathbf{x_0}) = \mathcal{N}(\mathbf{x_t};\sqrt{\bar{\alpha_t}}\mathbf{x_0}, (1 - \bar{\alpha_t})\mathbf{I})$$

It is clear that we can quickly sample a noisy image $\mathbf{x_t}$ at any timestep t. Given that we are using a conditional diffusion model, this allows us to randomly sample a timestep during training and feed the step, as well as the noise at said step, as the conditioned property during the forward diffusion process.

### The Reverse Process

To reverse the above noising process (i.e. $q(\mathbf{x_{t-1}}|\mathbf{x_t})$ ), we would need to know the entire dataset of possible noised images, which is essentially impossible. Thus, we seek to learn an _approximation_ of this reverse process. We will call this approximation $p_\theta$ . **(make note here about the beta size and why that allows us to treat this as a gaussian distribution [4])**

As we are describing the same distribution in the forward process, but in the opposite direction, we can use the same Markovian property to write:

<!-- Also, note that we can consider x_T to be "x_0" and traverse the process normally, just subbing back in our renamings -->

$$\begin{align}
p_{\theta}(\mathbf{x_{0:T}}) &= p_{\theta}(\mathbf{x_{T}}, \dots, \mathbf{x_{0}}) && \text{(\*)}\\
                             &= p(\mathbf{x_{T}}) p_{\theta}(\mathbf{x_{T-1}} | \mathbf{x_{T}}) \dots p_{\theta}(\mathbf{x_0} | \mathbf{x_1}) && \text{(Markov Property)} \\
                             &= p(\mathbf{x_T})\prod_{t=1}^{T} p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_T}) \\
\end{align}$$

(\*) Note that we do not condition $\mathbf{x_T}$ on anything here. **(explanation necessary [2])**

Also, we make mention here that conditioning our $\mathbf{\mu_{\theta}}(\mathbf{x_t}, t)$ on $\mathbf{x_0}$ in:

$$p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_t}) = \mathcal{N}(\mathbf{x_{t-1}}; \mathbf{\mu_{\theta}}(\mathbf{x_t}, t), \beta_t \mathbf{I})$$

,allows us to work with the reverse probabilities far more easily. **(Maybe include a more mathematical reasoning here [2])** We will rewrite this as:

$$q(\mathbf{x_{t-1}} | \mathbf{x_t}, \mathbf{x_0}) = \mathcal{N}(\mathbf{x_{t-1}}; \mathbf{\tilde{\mu_t}}(\mathbf{x_t}, \mathbf{x_0}), \tilde{\beta_t}\mathbf{I})$$

**(Leave this alone for now, but I believe we can switch to q since we are essentially now dealing with the forward process, as anything we are predicting comes from $\mathbf{x_0}$ and thus is always in the forward direction)**

**(Reorganize this)**

## Loss Function

Next, we seek to optimize the negative log-likelihood. We are, in essence, using a Variational Auto Encoder, wherein we want our 'probabilistic decoder' $p(\mathbf{x_{t-1}} | \mathbf{x_{t}})$ to closely approximate our 'probabilistic encoder' $q(\mathbf{x_{t-1}} | \mathbf{x_{t}})$ ([6]). To accomplish this, we need the ability to compare these two distributions. Thus, we use Kullback-Leibler (KL) divergence **(maybe make reference of this)** to achieve this comparison. It follows then that we want to minimize our KL divergence. However, we also should maximize the likelihood that we generate real samples, or $p_{\theta}(\mathbf{x_0})$. Conviently, we can employ Variational Lower Bounds (VLB)/Evidence Lower Bounds (ELBO) to achieve this concurrently **([7], [8])**. We will refer to Variational Lower Bounds as VLB for the remainder of this paper.

To derive VLB, we expand the above KL divergence:

$$\begin{align*}
D_{\text{KL}}(q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) || p(\mathbf{x_{1:T}} | \mathbf{x_{0}})) &= \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \\
    &= \int q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) \log{\Big ( \frac{q(\mathbf{x_{1:T}} | \mathbf{x_{0}})}{p(\mathbf{x_{1:T}} | \mathbf{x_{0}})}\Big )} d\mathbf{x_{1:T}} \\
    &= \int q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) \log{\Big ( \frac{q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) p_{\theta}(\mathbf{x_{0}})}{p(\mathbf{x_{1:T}}, \mathbf{x_{0}})}\Big )} d\mathbf{x_{1:T}} && \text{(Bayes' Rule)}\\
    &= \int q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) \Big \[ \log{p_{\theta}(\mathbf{x_{0}})} + \log{\Big ( \frac{q(\mathbf{x_{1:T}} | \mathbf{x_{0}})}{p(\mathbf{x_{1:T}}, \mathbf{x_{0}})}\Big )} \Big \] d\mathbf{x_{1:T}} \\
\end{align*}$$
<!-- This is super annoying, and will not let me combine these two align environments without generating an issue -->
$$\begin{align}
    &= \log{p_{\theta}(\mathbf{x_{0}})} \int q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) d\mathbf{x_{1:T}} + \int q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}}) \log \Big( \frac{q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}})}{p(\mathbf{x_{1:T}}, \mathbf{x_{0}})}\Big) d\mathbf{x_{1:T}}\\
    &= \log{p_{\theta}(\mathbf{x_{0}})}  + \int q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}}) \log \Big( \frac{q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}})}{p(\mathbf{x_{1:T}}, \mathbf{x_{0}})}\Big) d\mathbf{x_{1:T}} && \text{$\Big( \int f(x)dx = 1\Big)$}\\
    &= \log{p_{\theta}(\mathbf{x_{0}})}  + \int q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}}) \log \Big( \frac{q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}})}{p(\mathbf{x_{0}} | \mathbf{x_{1:T}}) p_{\theta}(\mathbf{x_{1:T}})}\Big) d\mathbf{x_{1:T}} && \text{(Bayes' Rule)}\\
\end{align}$$
<!-- Same problem again -->
$$\begin{align}
    &= \log{p_{\theta}(\mathbf{x_{0}})}  + \int q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}}) \Big \[ \log \Big( \frac{q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{1:T}})}\Big) - \log p_{\theta}(\mathbf{x_{0}} | \mathbf{x_{1:T}}) \Big \] d\mathbf{x_{1:T}} \\
    &= \log{p_{\theta}(\mathbf{x_{0}})}  + \int q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}}) \log \Big( \frac{q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{1:T}})}\Big) - q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}})\log p_{\theta}(\mathbf{x_{0}} | \mathbf{x_{1:T}}) d\mathbf{x_{1:T}} \\
    &= \log{p_{\theta}(\mathbf{x_{0}})}  + \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big \[ \log \Big( \frac{q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{1:T}})}\Big) - \log p_{\theta}(\mathbf{x_{0}} | \mathbf{x_{1:T}}) \Big \] \\
\end{align}$$

$$\begin{align}
    &= \log{p_{\theta}(\mathbf{x_{0}})}  + \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big \[ \log \Big( \frac{q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{1:T}})}\Big) \Big \] - \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big \[ \log p_{\theta}(\mathbf{x_{0}} | \mathbf{x_{1:T}}) \Big \] \\
    &= \log{p_{\theta}(\mathbf{x_{0}})}  + D_{\text{KL}}(q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) || p(\mathbf{x_{1:T}})) - \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big \[ \log p_{\theta}(\mathbf{x_{0}} | \mathbf{x_{1:T}}) \Big \] \\
\end{align}$$

Now, we have:

$$D_{\text{KL}}(q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) || p(\mathbf{x_{1:T}} | \mathbf{x_{0}})) = \log{p_{\theta}(\mathbf{x_{0}})}  + D_{\text{KL}}(q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) || p(\mathbf{x_{1:T}})) - \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big \[ \log p_{\theta}(\mathbf{x_{0}} | \mathbf{x_{1:T}}) \Big \]$$

If we rearrange this, we get:

$$ \log{p_{\theta}(\mathbf{x_{0}})} - D_{\text{KL}}(q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) || p(\mathbf{x_{1:T}} | \mathbf{x_{0}})) = \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big \[ \log p_{\theta}(\mathbf{x_{0}} | \mathbf{x_{1:T}}) \Big \] - D_{\text{KL}}(q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) || p(\mathbf{x_{1:T}}))$$

The left hand side of this expression yields the desirable relationship with $\log p_{\theta}(\mathbf{x_{0}})$, and thus is our VLB, which we will call $-L_{\text{VLB}}$. Noting that KL divergence is always non-negative:

$$\log{p_{\theta}(\mathbf{x_{0}})} - D_{\text{KL}}(q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) || p(\mathbf{x_{1:T}} | \mathbf{x_{0}})) \leq \log p_{\theta}(\mathbf{x_{0}})$$

As we seek to maximize the log-likelihood, and acknowledging that all maximizations can be expressed in terms of minimization, we once more rewrite the above expression in terms of said minimization:

$$-\log p_{\theta}(\mathbf{x_{0}}) \leq D_{\text{KL}}(q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) || p(\mathbf{x_{1:T}} | \mathbf{x_{0}})) - \log{p_{\theta}(\mathbf{x_{0}})} $$

We may further simplify this VLB:

$$\begin{align}
L_{\text{VLB}} &= D_{\text{KL}}(q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) || p(\mathbf{x_{1:T}} | \mathbf{x_{0}})) - \log{p_{\theta}(\mathbf{x_{0}})} \\
               &= \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big \[ \frac{q(\mathbf{x_{1:T}} | \mathbf{x_{0}})}{p(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big \] - \log{p_{\theta}(\mathbf{x_{0}})} \\
               &= \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big \[ \frac{q(\mathbf{x_{1:T}} | \mathbf{x_{0}})p_{\theta}(\mathbf{x_{0}})}{p(\mathbf{x_{1:T}}, \mathbf{x_{0}})} \Big \] - \log{p_{\theta}(\mathbf{x_{0}})} \\
               &= \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big \[ \frac{q(\mathbf{x_{1:T}} | \mathbf{x_{0}})}{p(\mathbf{x_{0}}, \mathbf{x_1}, \dots, \mathbf{x_{T}})} + \log p_{\theta}(\mathbf{x_{0}}) \Big \] - \log{p_{\theta}(\mathbf{x_{0}})} && \text{(Bayes' Rule)}\\
\end{align}$$ 

$$\begin{align}
              &= \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big \[ \frac{q(\mathbf{x_{1:T}} | \mathbf{x_{0}})}{p(\mathbf{x_{0:T}})} \Big \] +  \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \[ \log p_{\theta}(\mathbf{x_{0}}) \] - \log{p_{\theta}(\mathbf{x_{0}})} \\
              &= \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big \[ \frac{q(\mathbf{x_{1:T}} | \mathbf{x_{0}})}{p(\mathbf{x_{0:T}})} \Big \] + \log p_{\theta}(\mathbf{x_{0}}) - \log{p_{\theta}(\mathbf{x_{0}})} \\
              &= \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big \[ \frac{q(\mathbf{x_{1:T}} | \mathbf{x_{0}})}{p(\mathbf{x_{0:T}})} \Big \]\\
\end{align}$$
## Implementation

How we implemented it

## Results

what it accomplishes, how well it performs

## Installation

First clone the repository, and navigate to the colorized folder:

```sh
git clone git@github.com:Clement-W/MAT_180_ML_Projects.git
cd MAT_180_ML_Projects/colorized
```

To install the required libraries, there is 2 solutions. You can either install the requirements directly on your python installation, or install them into a virtual environement.

To install the requirements, you can just do:

```sh
pip3 install -r requirements.txt
```

To use a virtual environement:

* Install virtualenv if you don't already have it

```sh
pip3 install virtualenv
```

* Create a virtual environment

```sh
python3 -m venv colorized-env
```

* Activate the virtual environment

```sh
source colorized-env/bin/activate
```

* Install the required libraries into the environment
```sh
pip3 install -r requirements.txt
```

If you whish to leave the virtual environment, you can just do:
```sh
deactivate
```

## How to use our model

Walk through how a new user can use your model on their own data after cloning the repository

suggestion: create a notebook where the user can give an image, and the code to preprocess the image, run the model and everything is already included in it?

## Conclusion

feedback, improvements,...

## References

1. https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
2. https://arxiv.org/abs/1503.03585
3. http://www.stat.yale.edu/~pollard/Courses/251.spring2013/Handouts/Chang-MarkovChains.pdf
4. On the theory of stochastic processes (Feller Cornell Article )
5. [https://deeplearningbook.org](https://www.deeplearningbook.org/)
6. https://lilianweng.github.io/posts/2018-08-12-vae/#vae-variational-autoencoder
7. https://www.alexejgossmann.com/conditional_distributions/#FellerVol2
8. https://en.wikipedia.org/wiki/Evidence_lower_bound
<!-- #endregion -->
