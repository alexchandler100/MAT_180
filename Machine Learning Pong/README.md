# Pong AI

## Abstract

Group Members:
- Timothy Blanton

The goal of this project is to train a neural network to play pong. The game will be custom coded specifically for use within this project. The goal is to have two different play modes for the network:
1. Rally Mode
2. Competition Mode

In Rally Mode the network will attempt to pass the ball back and forth endlessly while in Compeition Mode the network will attempt to score the most amount of points.
The other player in the game will be represented by an algorithm that attemps to follow the ball's position but with a maximum speed compared to the network or a paddle that moves up and down at a constant speed.\
The neural network will be fed the y coordinate of its paddle, the x and y coordinate of the ball, and the difference between the y coordinate of its paddle and the ball. As an output it can accelerate the paddle upwards or downwards. The performance of the network will be measured by distance from the paddle to the ball, with a smaller distance giving it a better score, and the length of time it has survived.\
The network will be trained in generations. With each subsequent generation being based on the best performers in the previous one with some random changes to the weights. The trial will stop when the network misses the ball or the network hits the max amount of steps.

### Implementing the Game

The process for implementing the game is described in [Game](Game.ipynb) and also available as a script in [Game.py](Pong/Game.py). The file consists of 3 classes:
Pong, Paddle, and Game. The Pong is the object the paddles pass back and forth, the Paddle is the object the players use to interact with the Pong, and the Game is the object that contains the objects and helper functions to make the game run smoothly.
The detailed explanation for all the functions and variables each class contains as well as all code can be found in the [Game](Game.ipynb) notebook.

### Implementing the Network

The process for implementing the Network is described in [Network](Network.ipynb) and also available as a script in [Network.py](Network/Network.py). The file consists of two classes: Network and Layer.

The Layer class consists of a matrix of weights and vector of biases used to calculate the output of the layer. It also consists of a function to slightly modify the weights and biases based on a mutation rate and change value.

The Network class consists of an array of layers. It has a function that takes an input vector and outputs the feedforward ouput of the network. It also has a function that goes through each layer and modifies their weights and biases.

A more detailed explaination of the various functions and variables associated with each class can be found in the [Network](Network.ipynb) notebook.

## Rally Mode

The process of training the Network to rally is described in [Training](Training.ipynb) and also available as two scripts in [Evaluate.py](Network/Evaluate.py) and [Train.py](Network/Train.py).

This process consists of two main processes, the training process and the evaluation processes. 

The training process needs a generation size. It starts by generating that many randomly initialized neural networks. From there is runs the evaluation function on all of them, getting a fitness score back.
It then sorts the networks by their relative fitness. The best networks are copied until we have a generation worth of networks. From there each of the duplicated networks have their weights and biases slightly changed and the process is repeated.
Eventually the process reaches a stopping point and the weights and biases of the best network are outputed to a text file as well as the console output before it visually displays the best network's performance.

The evaluation process takes in a neural network and outputs its fitness score. In order to do this it goes through the game stepwise, getting the action the neural network would take and having its paddle take that action. 
At the end it returns the fitness score which is made by taking the number of game steps the neural network survived for and subtracing how far away the paddle was from the ball when it passed.

As shown in the [Training](Training.ipynb) notebook, this process returns relatively good results in a few generations.

## Competition Mode

Due to time constraints and frequent bugs, this mode was not able to be implemented.