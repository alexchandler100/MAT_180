# Pong AI

## Abstract

Group Members:
- Timothy Blanton

The goal of this project is to train a neural network to play pong. The game will be custom coded specifically for use within this project. The goal is to have two different play modes for the network:
1. Rally Mode
2. Competition Mode

In Rally Mode the network will attempt to pass the ball back and forth endlessly while in Compeition Mode the network will attempt to score the most amount of points.
The other player in the game will be represented by an algorithm that attemps to follow the ball's position but with a maximum speed compared to the network or a paddle that moves up and down at a constant speed.\
The neural network will be fed the x and y coordinates of the two paddles and the x and y coordinate of the ball. As an output it can accelerate the paddle upwards or downwards. The performance of the network will be measured by distance from the paddle to the ball, with a smaller distance giving it a better score, and every point scored representing a much larger point increase.\
The network will be trained in generations. With each subsequent generation being based on the best performers in the previous one with some random changes to the weights. The trial will stop when the network misses the ball.