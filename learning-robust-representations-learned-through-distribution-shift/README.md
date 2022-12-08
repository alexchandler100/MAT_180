Names of Group Members: Ayush Chakravarthy, Rishabh Jain, Essam Sleiman

Name of Project: Learning Robust Representations learned through distribution shift

1. We won’t be collecting any data, instead what we will do is compose certain popular vision datasets such as CIFAR-10, CIFAR-100, CelebA to form a ‘pre-training’ corpus.
2. We want to show the effect that pre-training and/or retrieval augmentation have on Computer Vision algorithms. More specifically, we want to evaluate whether having pre-training steps or retrieval mechanisms can positively or negatively affect the generalization ability of the learned representations.
3. We will be testing several pre-trained models such as ResNet-SimCLR and other SSL models along with training the backbones ourselves. The only novelty we may introduce is to implement a retrieval function to augment the posterior function from p(y|X) to p(y|X; {data}) where {data} is some data retrieved from an external dataset.
4. We will measure model performance through accuracy on the test dataset.

For M = {ResNet-18, AlexNet, Perceiver}
So, the objective is to measure the training loss and the testing accuracy on a downstream classification task for the following:
1. Vanilla M (would be the baseline)
2. M + Retrieval Augmentation
3. M + Pretraining on our corpus
4. M + Pretrained with SimCLR on ImageNet


# MAT 180 Machine Learning Group Projects

This repository contains group projects for students enrolled in MAT 180: The Mathematics of Machine Learning, in the Fall 2022 quarter at UC Davis. 

## Config file

1. Create a new conda environment
2. Install latest version of Pytorch

```python

$ pip install -r requirements.txt 
$ chmod +x run_perceiver_baseline.sh 
$ ./run_perceiver_baseline.sh 

```
 
 
## Background
### ResNet

A Convolutional Neural Net (CNN) is a clas of neural networks that specializes in processing data that has a grid-like topology, such as an image. A CNN typically has three layers: a convolutional layer, a pooling layer, and a fully connected layer. The convolution layer is the core building block of the CNN. This layer performs a dot product between two matrices, where one matrix is the set of learnable parameters otherwise known as a kernel, and the other matrix is the restricted portion of the receptive field. This means that, if the image is composed of three (RGB) channels, the kernel height and width will be spatially small, but the depth extends up to all three channels. During the forward pass, the kernel slides across the height and width of the image-producing the image representation of that receptive region. This produces a two-dimensional representation of the image known as an activation map that gives the response of the kernel at each spatial position of the image.

In theory, larger models should perform better, however, deep networks are hard to train because of the notorious vanishing gradient problem. As the gradient is backpropagated to earlier layers, repeated multiplication may make the gradient infinitely small. As a result, the deeper the network goes, the more its performance becomes saturated or even starts rapidly degrading. This problem of training very deep networks has been alleviated with the introduction of ResNet or residual networks and these Resnets are made up from Residual Blocks.

The method of bypassing the data from one layer to another is called as shortcut connections or skip connections. This approach allows the data to flow easily between the layers without hampering the learning ability of the deep learning model. The advantage of adding this type of skip connection is that if any layer hurts the performance of the model, it will be skipped. As a result, the vanishing gradient problem is mitigated and deep networks are able to be successfully trained with high success. 


### Perceiver

In a general Transformer, our first input data goes through a multi-headed attention layer through a feed-forward layer. In Multi headed attention there would be multiple self-attention layers. In the self-attention layer, each of the elements in the input is associated with all other words in the input. For Language tasks, if there are M words in the input sequence, then because of self-attention the space complexity becomes quadratic. Inside self-attention, we feed the input to three distinct fully connected layers to create query key and value vectors. The queries and keys undergo dot product matrix multiplication to produce a scoring matrix. The score matrix determines how much focus should a word be put on other words. Then we add softmax to get the highest probability values and this helps our model to provide more confidence on which words to attend to. And then it gets multiplied to value vector to get output vector. Because of multiplication between query and Key, Transformers in general attention instantly shoots upwards in space as input increases. For language tasks, it was fine as at maximum the input sequences would be batched to around 1000 or so. However, when considering images, our input sequences would be all the pixels in the image. For a simple image of 256x256, our pixels would be around 62,500 which is our input sequence and now our space complexity is 65536x65536. Several approaches came to provide alternative solutions such as VIT Transformers, where images have been broken into batches and then fed them however still it doesn’t solve our quadratic complexity.

The perceiver architecture generally tries to reduce this space complexity to a limit such that it should not be quadratic. To solve this issue, they have added a cross attention layer between the input sequence and multi-headed attention. In attention where we perform matrix multiplication between Query and Key where both were of size MxM where m is input sequence, in cross attention, our Query would be of size N where N< M. Using this our space complexity reduces to MxN this query of size N is called a latent array.


## Approach

We integrate various forms of retrieval into the pre-discussed architectures. The motivation behind this idea is from the notion of Episodic Memory in Cognitive Science and Psychology. Imagine a scenario where a student first learned to solve probability problems in MAT135A. Now, after a year they take MAT135B, where they encounter the same types of problems but with a Markov Chain flavor. In this scenario, their brain is able to actively retrieve encoded information from the previous experience of MAT135A, while solving MAT135B problems.
This exactly is the notion we are trying to integrate into our models. We use two styles of retrieval: 1. sampled 2. k-nearest neighbor.
In the sampled strategy, we randomly sample tensors from a database of previous 'experiences'. Note that these tensors are the representations outputted by the second to last layer of a pre-trained ResNet backbone trained using BYOL self-supervision signal.
However, in the k-nearest neighbor retrieval strategy we do something 'smarter'. We get the k-nearest tensors from the database of 'experiences'. This way, we can find the most similar 'experiences' from the store of experiences.

We use simple Cross Attention using these 'retrieved' representations to augment the learning process.


## Experiments

-- Perceiver noisy figure --

In this experiment, we run the experiment for noisy test set using knn retrieval augmented Perceiver against a baseline Perciever with No Retrieval.

-- ResNet Results --

In the two above experiments, we don't apply the noise augmentation to the Test Set, but ablate over the size of the retrieval buffer with knn strategy vs sampled strategy.

Pre-trained model: https://drive.google.com/drive/folders/1TsJTqHOQwuFZJdLr-ZAUQUwTh-fDPj1T?usp=sharing


## Results

From the results, we can infer that, at minimum, retrieval augmented learners show greater sample efficiency than without. In case of the Perceiver, we see that retrieval augmentation not only implies sample efficient learning but also this solves the problem of the data-hungry nature of attention based models. Furthermore, on further stress-testing the model (adding randomly sampled gaussian noise to the test images) we still see robustness from the Retrieval-Augmented Perceiver. 

The story for non attention based models (ResNet) is different though. We marginally underperform the baseline even though we see the same sample efficient learning curve as compared to vanilla ResNet. 
Finally, we are able to conclude that retrieval is indeed quite a understudied and can be used to learn robust representations. However, the case against retrieval is that empiricism. We essentially use more data per model forward pass which can be interpreted as an 'unfair' comparison against the baselines. We also don't incorporate any novel inductive biases into the model to prompt stronger forms of generalization rather just use more data.


![Alt text](/Screenshot_2022-12-07_at_8.12.12_PM.png)
Figure 1: Results show method accuracy over epoch. 

![Alt text](/Screenshot_2022-12-07_at_8.13.32_PM.png)
Figure 2: Performance impact by of k.

![Alt text](/Screenshot_2022-12-07_at_8.13.39_PM.png)
Figure 3: Cifar10 experiments.


Note: The authors would like to apologize for the unkempt codebase. There may be portions that are redundant or not necessarily useful. Further, we were a bit sloppy with our experimentation.













