# MAT 180 Machine Learning Group Projects

This repository contains group projects for students enrolled in MAT 180: The Mathematics of Machine Learning, in the Fall 2022 quarter at UC Davis. 
 
## Background
### ResNet

A Convolutional Neural Net (CNN) is a clas of neural networks that specializes in processing data that has a grid-like topology, such as an image. A CNN typically has three layers: a convolutional layer, a pooling layer, and a fully connected layer. The convolution layer is the core building block of the CNN. This layer performs a dot product between two matrices, where one matrix is the set of learnable parameters otherwise known as a kernel, and the other matrix is the restricted portion of the receptive field. This means that, if the image is composed of three (RGB) channels, the kernel height and width will be spatially small, but the depth extends up to all three channels. During the forward pass, the kernel slides across the height and width of the image-producing the image representation of that receptive region. This produces a two-dimensional representation of the image known as an activation map that gives the response of the kernel at each spatial position of the image.

In theory, larger models should perform better, however, deep networks are hard to train because of the notorious vanishing gradient problem. As the gradient is backpropagated to earlier layers, repeated multiplication may make the gradient infinitely small. As a result, the deeper the network goes, the more its performance becomes saturated or even starts rapidly degrading. This problem of training very deep networks has been alleviated with the introduction of ResNet or residual networks and these Resnets are made up from Residual Blocks.

The method of bypassing the data from one layer to another is called as shortcut connections or skip connections. This approach allows the data to flow easily between the layers without hampering the learning ability of the deep learning model. The advantage of adding this type of skip connection is that if any layer hurts the performance of the model, it will be skipped. As a result, the vanishing gradient problem is mitigated and deep networks are able to be successfully trained with high success. 


### Perceiver

In a general Transformer, our first input data goes through a multi-headed attention layer through a feed-forward layer. In Multi headed attention there would be multiple self-attention layers. In the self-attention layer, each of the elements in the input is associated with all other words in the input. For Language tasks, if there are M words in the input sequence, then because of self-attention the space complexity becomes quadratic. Inside self-attention, we feed the input to three distinct fully connected layers to create query key and value vectors. The queries and keys undergo dot product matrix multiplication to produce a scoring matrix. The score matrix determines how much focus should a word be put on other words. Then we add softmax to get the highest probability values and this helps our model to provide more confidence on which words to attend to. And then it gets multiplied to value vector to get output vector. Because of multiplication between query and Key, Transformers in general attention instantly shoots upwards in space as input increases. For language tasks, it was fine as at maximum the input sequences would be batched to around 1000 or so. However, when considering images, our input sequences would be all the pixels in the image. For a simple image of 256x256, our pixels would be around 62,500 which is our input sequence and now our space complexity is 65536x65536. Several approaches came to provide alternative solutions such as VIT Transformers, where images have been broken into batches and then fed them however still it doesn’t solve our quadratic complexity.

The perceiver architecture generally tries to reduce this space complexity to a limit such that it should not be quadratic. To solve this issue, they have added a cross attention layer between the input sequence and multi-headed attention. In attention where we perform matrix multiplication between Query and Key where both were of size MxM where m is input sequence, in cross attention, our Query would be of size N where N< M. Using this our space complexity reduces to MxN this query of size N is called a latent array.


### Retrieval

###KNN


####Regular Retrieval (I forgot the name Ayush add)

## Approach

## Experiments

## Results

Todo: Writeup about Perceiver and Resnet
Baseline: on wandb
Retrieval: Write about Sample Retrieval: 








