# Car Brand Classifier
###### Created by Jaime Luna, Nicholas Stein

Proposal

Data Source:
We have downloaded a file of photos of car logo emblems from Kaggle. 
Included with the file are labels for the brand of each car in the photos.

Task:
We will aim to identify the brand of the vehicles from the images containging logos. 

Model:
We will use a Multi-class classification Neural Network beginning with the minimum 3 layers.
ReLU activation will be contained in the middle layer and softmax in the classification layer.
Considering this is a more complicated image than the digit classifier even with the black and white
images we intend to use, we will add the necessary complexity to the network to improve the
poor result we expect. Adding a convolution layer will also be considered, perhaps using ready to use 
libraries if necessary.

Evaluation:
We will separate data into training, validation, and test set to determine accuracy. Since data set is labeled, 
accuracy will be apparent from output. 
