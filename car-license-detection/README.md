# MAT-180-final-project

# Project Proposal

## 1. Include names of every group member
      name: Yaoyuan Shi
            Yixue Zhao

    
## 2. Name of Project
      Chinese Car license detection


## 3. Outline goals of the project i.e. indicate the following:
      
    (a) How will the data be collected?
        from https://github.com/detectRecog/CCPD
        
    (b) What task do you want to accomplish with the data?
        recongnize the numbers and characters on the image of car license
        
    (c) What kind of learning algorithm do you propose using to accomplish this task?
        We going to use Convolutional Network to recognize and analyze the numbers and characters in the pictures and transform into the data accordingly. 
    
    (d) How will you measure your performance of the task?
        If we pick ar random photo from the dataset, and it can be recognize and detect as an output
        
## Requirement
numpy,
albumentations,
Pytorch,
OpenCV,
Glob,
In order to increase the speed, we will use gpu to run



## Datast

The **Chinese City Parking Dataset** (**CCPD**) is a dataset for license plate detection and recognition. It contains  over 250k unique car images, with license plate location annotations.


## Model

We used a convolutional neural network for recognition, the model can input the license plate and output all recognized characters.

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

## Environment

Developing with Python and Pytorch requires running on a GPU device.
