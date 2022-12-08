# Car Brand Classifier
Created by Jaime Luna, Nicholas Stein

#### Proposal

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

### Using this project
##### Data
In order to use alternate datasets, the user will need to provide two folders. These folders correspond to the training set and the test set. Both folders will contain identically labeled folders that contain images. For this project, the labels correspond to car brands, and the images are car brand logos. The images should be .jpg file type.

##### Required Input
In order for the project to read and process the images, the user will need to edit the 'train_dir' and 'test_dir' strings in the Initializing_Images.ipynb notebook. 'train_dir' will be the directory of the training data folder, and 'test_dir' will be the directory of the test data folder. Once the directories have been specified, running the notebook will rename every image for the sake of iterability. The files in each folder will now be named 'i.jpg' with 0 <= i <= (number of images in the folder).

Next, the image_upload.ipynb notebook will be run in order to process each image.

Following this, the predictor.ipynb notebook will be run to predict the car brand based on the logos in the images.

##### Performance
We were unable to accurately predict the car brands by their logos.
