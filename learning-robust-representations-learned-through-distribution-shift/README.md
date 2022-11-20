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
