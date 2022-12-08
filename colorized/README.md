<!-- #region -->

# COLORIZED - Colorization Of Lost Or Ruined ImageZ Employing Diffusion

Clément Weinreich

Brennan Whitfield

Vivek Shome

<details>
<summary>Table of Contents</summary>
    <ul>
        <li> <a href="#Introduction">Introduction</a></li>
        <li> <a href="#Project_Management"> Project Management </a> </li>
        <li> <a href="#Instructions"> Instructions </a></li>
        <li> <a href="#Theory">Theory </a> </li>
        <details> 
            <summary> <a href="#Implementation"> Implementation </a> </summary>
            <ul> 
                <li> <a href="#Diffusion_Basics"> Diffusion Basics </a></li>
                <li> <a href="#Forward_Process">Forward Process</a></li>
                <details> <summary> <a href="#Neural_Network">Neural Network</a> </summary>
                    <ul>
                        <li><a href="#UNet_Overview">Overview of the U-Net architecture</a> </li>
                        <li><a href="#Timestep_Embedding">Timestep Embedding</a></li>
                        <li><a href="#Conditional_input">Conditional Image Input</a></li>
                        <li><a href="#Core_Convolution_Block">Core Convolution Block</a></li>
                        <li><a href="#Resnet_Block">Resnet Block</a></li>
                        <li><a href="#Sample_Blocks">Up/Down Sampling Blocks</a></li>
                        <details> <summary> <a href="#Custom_Architecture">Our Custom UNet Architecture</a> </summary>
                            <ul>
                                <li><a href="#Encoder">Encoder</a></li>
                                <li><a href="#Bottleneck">Bottleneck</a></li>
                                <li><a href="#Decoder">Decoder</a></li>
                            </ul>
                        </details>
                    </ul>
                </details>
                <li><a href="#Reverse_Process">Reverse Process</a></li>
                <li><a href="#Training">Training</a></li>
            </ul>
            </details>
        <li><a href="#Results">Results</a></li>
        <li><a href="#Discussions">Discussions</a></li>
        <li><a href="#Installation">Installation</a></li>
        <li><a href="#Model_How_To">How to Use Our Model</a></li>
        <li><a href="#Conclusion">Conclusion</a></li>
        <li><a href="#References">References</a></li>
    </ul>
</details>

## Instructions (delete this once finished)
    
Your README.md file (located at the root of your project directory) should completely describe the project, what it accomplishes, how well it performs, and walk through how a new user can use your model on their own data after cloning the repository. This means, by following instructions in your README.md file, I should easily be able to run your algorithm on a new example not in your dataset and see that the model accomplishes what it promised to accomplish (at least up to your claimed performance measure).

<div id="Introduction"></div>

## Introduction

The real world problem we sought to solve was the colorization of greyscale images found in the Shields library archive here in Davis. This project intended to provide the library with a means of viewing a local landmark/area from the perspective of an individual present at the original time. To achieve this, we decided to employ an image-to-image conditional diffusion model, which would be trained on a dataset ([Image Colorization on Kaggle](https://www.kaggle.com/datasets/shravankumar9892/image-colorization/code)) of 50 thousand images, consisting of 25 thousand 224 x 224 greyscale-color image pairs. 

As the information on this direct concept is fairly sparse, we took it upon ourselves to seek an understanding of the model at every level. Moreover, we recognized the value of comprehending all mathematical theory involved (as this is a math course), and have thus detailed below the theory, as well as the steps we took to grasp each idea. We would like to emphasize the amount of time which went into both the writing, as well as the thought, behind each step in the process. This project was quite ambitious, so we additionally acknowledge the tradeoff made between performance of the model, and the depth in which we understood it.

In the following sections, we dissect why a conditional diffusion model works, the mathematics behind it, and the implementation of said model.

**LEAVE FOR NOW (until we talk about FID)**
###### How will you measure your performance of the task?

To measure the performance of the task, we will use the [Fréchet Inception Distance](<[https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance)>) (FID). The FID metric compares the distribution of generated images with the distribution of a set of real images (ground truth). The FID metric was introduced in 2017 and is the current standard metric for assessing the quality of generative models.

<div id="Project_Management"></div>

## Project Management

<div id="Instructions"></div>


## Instructions

how to run the project

<div id="Theory"></div>

## Theory

Because of an issue regarding big equations in github markdown, all the theory part is written in Theory.md, and exported in html as Theory.html. This section is an important part of the project as we spent many days to understand and derive all the expressions, in order to have our loss function.
<!-- #endregion -->

<!-- #region -->

<div id="Implementation"></div>

## Implementation

```
you may argue to use a build-in implementation if the algorithm involved is
overly complicated but necessary to the task. After solving the task with your own
implementation, you may also, in a separate notebook, use a built-in implementation
and compare performance.
```

As our project is quite ambitious, the most complex part was to understand how diffusion models work. This is why, most of our work and of our research is explained in the theory section. Concerning the implementation, most of the papers we are reffering to, published their code in github ([1],[2]). As exploring and understanding the theory behind diffusion models took us 3 to 4 weeks in total, we have chosen to draw inspiration from these github repository, as well as articles going through these implementations like [the one from Hugging Face](https://huggingface.co/blog/annotated-diffusion). Even though we got inspired from these codes, we can note that some algorithms had to be modified to adapt to conditional diffusion.

As our project requires to use a neural network with an encoder-decoder architecture with complex types of layers, we are using the python library PyTorch. Thus, most of the objects that we are manipulating are not numpy arrays, but PyTorch tensors. These tensors behaves like numpy arrays, but are more convenient to track the operations and therefore compute the gradients required by the backpropagation step. To do so, PyTorch uses automatic differenciation, which consists of building an implicit computational graph when performing mathematical operations. By knowing the derivative of elementary mathematical operations, it is then possible to compute efficiently the gradient of every functions. 

<div id="Diffusion_Basics"></div>

### Diffusion basics

In the theory section, we often make reference to $\bar{\beta}_t$ or $\bar{\alpha}_t$. To compute the noise scheduler, we implemented two different schedules:
- The cosine schedule which is implemented the same way as proposed in [2]:
$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, f(t) = cos\Big(\frac{t/T + s}{1 + s}  \frac{\pi}{2}\Big)^2$$
$$\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}$$
- The linear schedule which is implemented the same way as proposed in [1], with the same constants. Thus, it just a list of evenly spaced numbers starting from 0.0001 to 0.02.

In order to improve the speed of the algorithms, the values of many constants are computed, and stored in the class `ConstantDiffusionTerms`. This consists of lists containing all the values from $t=0$ to $t=T$ of:
$$\beta_t, \alpha_t, \bar{\alpha}_t, \bar{\alpha}_{t-1}, \frac{1}{\sqrt{\alpha_t}}, \sqrt{\bar{\alpha}_t}, \sqrt{1 - \bar{\alpha}_t}, \tilde{\beta}_t, \sqrt{\tilde{\beta}_t}$$

Thanks to the method `extract_t_th_value_of_list`, we can then extract the $t$ element of these lists, with respect to the batch size.

<div id="Forward_Process"></div>

### Forward process

As all the closed forms are computed, it is now easy to implement the forward process, which corresponds to 
$$q(\mathbf{x_t} | \mathbf{x_0}) = \mathcal{N}(\mathbf{x_t};\sqrt{\bar{\alpha_t}}\mathbf{x_0}, (1 - \bar{\alpha_t})\mathbf{I})$$
or using the reparametrization trick
$$\mathbf{x_t} = \sqrt{\bar{\alpha_t}}\mathbf{x_0} + \sqrt{1 - \bar{\alpha_t}}\mathbf{\epsilon}$$

Thus, we only need to compute a random noise $\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$ of the same size as the image, and compute $\mathbf{x_t}$ thanks to the closed forms computed earlier.

<div id="Neural_Network"></div>

### The neural network

<div id="UNet_Overview"></div>

#### Overview of the U-Net architecture

The most complex part of the implementation, is designing the neural network. This neural network has an encoder-decoder architecture as used in the Variational Auto Encoder. More precisely, the U-Net architecture was used in [1], and offers good results. The same type of architecture was used in [2], [3] and [4]. Here is the idea of this architecture:

<img src="assets/u-net-architecture.png" alt="unet architecture picture" style="height: 400px" align="center"/>
<br>
<div style='text-align:center'> <b>Figure 1</b>: Base U-Net architecture </div>


This architecture first downsamples the input in term of spatial resolution, but with more and channels because of the convolutions (64 -> 128 -> 256 -> 512 -> 1024). It also has a bottleneck in the middle that allow the network to learn the most important information to perform the task. Then, an upsampling is performed in order to output an image of the same shape as the input. Between the layers of identical size of the encoder and decoder, there is also residual connections (or skip connections), which improves the gradient flow.

In recent work on Diffusion models ([1],[2],[3],[4]), it has been shown that the self-attention layers greatly improves the results of diffusion models. In this first implementation, we made the choice of not implementing attention layers. As we don't have the computing power to train a model with hundreds of millions of parameters, we also reduced the size of the network in terms of depth and number of convolution kernels.

In this section, each main blocks are described, in order to present the architecture of our custom U-Net.

<div id="Timestep_Embedding"></div>

#### Timestep embedding

As seen in the theory part, the neural network must also take the timestep $t$ (indication on the noise level) as input. The authors of [1] employed the sinusoidal position embedding to encode $t$. The sinusoidal position embedding has been proposed by the famous Transformer architecture [5]. Sinusoidal positional embedding aims to generate an embedding that take into account the order of the information, using the sin and cos functions. So it takes a integer $t$ as input, and output a vector in $\mathbb{R}^d$, $d$ being the desired embedding dimension.

The sinusoidal positional embedding is defined by a function $f: \mathbb{N} \to \mathbb{R}^d$:

$$f(t)^{(i)} = \begin{cases}
      sin(\omega_k t), & \text{if}\ i=2k \\
      cos(\omega_k t), & \text{if}\ i=2k+1
    \end{cases}$$ 

$$\text{For } 1 \leq i \leq d, \forall k \in \mathbb{N} \text{ and } \omega_k = \frac{1}{1000^{2k/d}}$$
    
This positional encoding forms a geometric progression from $2\pi$ to $10000 \cdot 2\pi$. In the end, we obtain a vector containing pairs of sin and cos for each frequency. This also implies that $d$ must be divisible by 2.

<div id="Conditional_input"></div>

#### Conditional image input

The dataset consists of input-output image pairs $\{z_i,x_i\}^N_{i=1}$ where the samples are sampled from an unknown conditional distribution $p(x|z)$. Here, $x$ is the color image and $z$ is the grayscale image. This conditional input $z$ is given to the denoising model $\epsilon_\theta$, in addition to the noisy target image $x_t$ and the timestep $t$. In practice, to condition the model we followed the work of [4]. The input $z$ is concatenated with $x_t$ along the channel dimension before entering the first layer of the U-Net. 

<div id="Core_Convolution_Block"></div>

#### Core convolution block

The core convolution block is the elementary block of the bigger blocks described below. This convolution block simply apply a 2D convolution with kernels of size 3x3 and a padding of 1. A group normalization is applied to the output of the convolution, and then the SiLU (Sigmoid Linear Unit) activation function is applied.

<div id="Resnet_Block"></div>

#### Resnet Block

In our implementations, we have chosen to use the residual block as employed by [1]. The resnet block take as input an image but also the timestep embedded with sinusoidal embedding. At first, the input image goes through the first core convolution block. At the same time, the embedded timestep is sent through a one layer perceptron that output a vector of the same size as the number of kernels applied in the first core convolution block. Then, the timestep is reshaped in order to be compatible in number of dimensions with the output of the first core convolution block. This allows us to add the output of the MLP, with the output of the first convolution block. The result is then sent through another core convolution block.

The last step consists in adding the residual connection, which corresponds to add the input image of the resnet block, with the last result. If the case where the channel dimension is incompatible between those two, we apply a convolution with a kernel of size 1 in order to make them compatible.

<div id="Sample_Blocks"></div>

#### Up-sample and Down-sample blocks

As we can see in Figure 1, a downsample and an upsample operation is applied after 2 main blocks of convolutions (which are resnet blocks in our case). For the downsample operation, we are using a convolution layer with a kernel of size 4x4, a stride of 2 and a padding of 1. For the upsample operation, we use a transposed convolution layer with a kernel of size 4x4, a stride of 2 and a padding of 1. A transposed convolution generate an output feature map, with has a greater spatial dimension than the input feature map. The transposed convolution broadcasts each input elements via the kernel, which leads to an output that is larger than the input. 

We can express the convolution operation as a matrix multiplication between a sparse matrix containing the information of the kernel $W_{sparse}$, and a column vector which is the flattened (by row) version of the input matrix $X_{flattened}$. This result of this operation gives a column vector $Z_{flattened}$ which can then be reshaped to produce the same result as a classic convolution operation $Z$. Now, if we take an input matrix that has the same shape of $Z$, and perform a matrix multiplication with the transposed sparse kernel matrix $W_{sparse}^T$, we obtain a result that has the same shape as $X_{flattened}$. Then, we just have to reshape it to produce a result of the same shape as $X$. Thus, we performed an upsampling of $Z$. By increasing the stride and the kernel size, we can generate outputs of greater sizes.


<div id="Custom_Architecture"></div>

#### Architecture of our custom U-Net

Now that we defined the main building blocks of our custom U-Net, let's expand its architecture. The following table does not include the MLP of the sinusoidal position embedding as it is done at first only for the timestep $t$. As mentioned previously, our U-net is much smaller than the other u-net employed for state of the art diffusion models. In the end, we get a network having only ~6.9 Millions of parameters.

<div id="Encoder"></div>

##### Encoder

| **Layer**          	| **No Input channels** 	| **No Output channels** 	|
|--------------------	|:---------------------:	|------------------------	|
| Conv2d            	| img_channels          	| 32                     	|
| ResNet block       	| 32                    	| 64                    	|
| ResNet block       	| 64                    	| 64                    	|
| Conv2d (downsample)  	| 64                    	| 64                    	|
| ResNet block       	| 64                    	| 128                    	|
| ResNet block       	| 128                   	| 128                    	|
| Conv2d (downsample) 	| 128                   	| 128                    	|
| ResNet block       	| 128                   	| 256                    	|
| ResNet block       	| 256                   	| 256                    	|

   The encoder downscale the image while adding more and more feature maps.


##### Bottleneck

| **Layer**          	| **No Input channels** 	| **No Output channels** 	|
|--------------------	|:---------------------:	|------------------------	|
| ResNet block       	| 256                    	| 256                    	|
| ResNet block       	| 256                   	| 256                    	|
    
The bottleneck is the part of the network with the lowest image dimension, which compress all the important information of the image.

<div id="Decoder"></div>

##### Decoder

| **Layer**          	| **No Input channels** 	| **No Output channels** 	|
|--------------------	|:---------------------:	|------------------------	|
| ResNet block       	| 256                    	| 128                    	|
| ResNet block       	| 128                    	| 128                    	|
| ConvTranspose2D (up) 	| 128                    	| 128                    	|
| ResNet block       	| 128                    	| 64                    	|
| ResNet block       	| 64                    	| 64                    	|
| ConvTranspose2D (up) 	| 64                    	| 64                    	|
| ResNet block       	| 64                    	| 32                    	|
| ResNet block       	| 32                    	| 32                    	|
| ResNet block       	| 32                    	| 32                    	|
| Conv2d            	| 32                    	| img_channels             	|

The decoder upsample the input features from the bottlenck, in order to output an image of the same size as the input in terms of spatial dimenions, and number of channels.

<div id="Reverse_Process"></div>

### Reverse process

The reverse process (or 'inference' for our purposes) follows exactly from Algorithm 2 **LINK TO ABOVE ALGORITHM** of the Theory section above. We begin by sampling a completely noisy image $\mathbf{x_T}$ from a Gaussian distribution $\mathcal{N}(\mathbf{0},\mathbf{I})$. As our goal is to subtract noise from this image until we have the desired $\mathbf{x_0}$, at each timestep $0 \leq t \leq T$, we begin by sampling random noise $\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. Next, since each step can be seen as calculating $\mathbf{x_{t-1}}$ from $\mathbf{x_t}$ **LINK THIS TO COMMON TERMS SECTION**, we perform the reparameterization trick to achieve:

$$\mathbf{x_{t-1}} = \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x_t} - \frac{1 - \alpha_t}{\sqrt{1-\bar{\alpha}_t}} \mathbf{\epsilon}_\theta (\mathbf{x_t}, \mathbf{z},t) \Big) + \sqrt{\beta_t}\mathbf{\epsilon}$$

Thanks to our ```ConstantDiffusionTerms``` class, many of these closed forms are already computed.

Thus, we use our model $\mathbf{\epsilon}_{\theta}$ to predict the noise between steps, and feed this predicted noise into the reparameterization of the predicted mean **THIS CAN ALSO BE FOUND IN THEORY ABOVE**:

$$\mu_{\theta}(\mathbf{x_t}, t) = \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x_t} - \frac{1 - \alpha_t}{\sqrt{1-\bar{\alpha}_t}} \mathbf{\epsilon}_\theta (\mathbf{x_t}, \mathbf{z},t) \Big)$$

This gives us $\mathbf{x_{t-1}}$. Repeating this process T times yields us $\mathbf{x_0}$, which is a colorized version of the greyscale input.

<div id="Training"></div>

### Training

To train this model, we look to Algorithm 1 **LINK TO ABOVE**. As diffusion models seek to produce images closely correlated with the data they are trained on, we sample a color-greyscale image pair $(\mathbf{x_0}, \mathbf{z})$ from our data distribution $p(\mathbf{x}, \mathbf{z})$. Then, we sample a random timestep t from a uniform distribution of timesteps $\{ 1, \dots, T \}$, and as discussed in the theory section above, we can use the Markov Property to produce a noisy image $\mathbf{x_{t}}$ at said timestep:

$$\mathbf{x_t} = \sqrt{\bar{\alpha}_t}\mathbf{x_0}+\sqrt{1 - \bar{\alpha}_t}\epsilon$$

As before, we need to sample $\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$ to perform this. 

Now, we take a gradient descent step using our loss function:

$$\nabla_\theta \| \mathbf{\epsilon} - \mathbf{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x_0}+\sqrt{1 - \bar{\alpha}_t}\epsilon, \mathbf{z} ,t) \|^2$$

We repeat this until converged. Note, again, that our loss function is essentially the normed-difference between the theoretical noise, and our noise predicting model.

<div id="Results"></div>

## Results

We trained the network for 5 epochs with 2000 images (see `notebook/train.ipynb`) in order to be able to produce results. The results of our project can be found in the notebook in the folder `demo`. First, the forward process is working well. Here is the forward process using the linear schedule on 300 timesteps ($T = 300$):
<img src="assets/linearschedule.png" alt="Linear schedule" style="width: 800px" align="center"/>
<br>
<div style='text-align:center'> <b>Figure 2</b>: Forward process with linear schedule </div>

Now the results of the reverse process are not satisfying. This not suprising as the network has only be trained for 5 epochs. Here, we try to colorize this image:

<img src="assets/grayscale.png" alt="Grayscale image" style="width: 100" align="center"/>
<br>
<div style='text-align:center'> <b>Figure 3</b>: Grayscale image that we seek to colorize </div>


Then, here is 10 evenly separated noisy images from $\mathbf{x_T}$ to $\mathbf{x_0}$, using $T=300$:

<img src="assets/reverse.png" alt=" image" style="width: 800" align="center"/>
<br>
<div style='text-align:center'> <b>Figure 4</b>: Reverse process</div>

We notice that the noisy image is changing across the timesteps. We also notice some sort of structure in the noisy image, which slightly resembles the branch on the right of the image. Finally, this model output a (non accurate) colorized version of the grayscale image.

<div id="Discussions"></div>

## Discussions

As stated previously, we spent most of the time doing research and working on the theory side of conditional diffusion models. Thus, the time left for the implementation part was shorter as expected. It was still important for us to have the whole pipeline working, so the project can be improved by adding features. This is why we trained a small model on a small amount of data. A first limit of this training is that only 2000 images was used. This is mostly due to a lack of RAM from our computers, but we also wanted to use a subset of the dataset to make the training faster. We also did not created a test and validation set, as we knew that results won't be interpretable. At first, we planned to evaluate the model using the FID (Fréchet inception distance) score, which is a metrics used to evaluate generative models. For lack of time, we decided to not implement the `eval` method present in `scripts/eval.py`, and focus on this readme file. Finally, the results we obtained are far from being the colorized input image. As the entire pipeline is already written, the model needs to be improved, and the dataset needs to be better handled. Thereafter, we could expect better results.

We can also note that, diffusion models are a very recent research topic, we are working on research papers from 2 years ago or even from 2022. This makes the project really challenging when it comes to research.

....

<div id="Installation"></div>

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

<div id="Model_How_To"></div>

## How to use our model

Walk through how a new user can use your model on their own data after cloning the repository

suggestion: create a notebook where the user can give an image, and the code to preprocess the image, run the model and everything is already included in it?

<div id="Conclusion"></div>

## Conclusion

feedback, improvements,...

<div id="References"></div>

## References

[1] DDPM https://arxiv.org/pdf/2006.11239.pdf + associated github (https://github.com/lucidrains/denoising-diffusion-pytorch)


[2] IDDPM (https://arxiv.org/pdf/2102.09672.pdf) + associated github (https://github.com/openai/improved-diffusion)


[3] Palette: Image-to-Image Diffusion Models (https://arxiv.org/pdf/2111.05826.pdf)


[4] Image Super-Resolution via Iterative Refinement https://arxiv.org/pdf/2104.07636.pdf


[5] Attention is all you need https://arxiv.org/abs/1706.03762
<!-- #endregion -->
