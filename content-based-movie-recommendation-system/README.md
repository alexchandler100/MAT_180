# MAT180_2022Fall_Final
# Learning to Play Pong

<!-- ABOUT THE PROJECT -->
## Objective
In this project, we will be training a model to learn to play Pong without human knowledge of the game. We will be using the OpenAI gym to run and interact with the Pong Atari game and the deep Q-learning algorithm. 

## Overview
As pre-trained model is provided, we will make modifications to improve its ability to learn and train the provided pre-trained model to acceptable performance. The model will use its Q network to decide what moves to take when playing the game. This network will push each decision it made to an experience buffer. This replay buffer will be polled from to make updates to the Q network using the loss function. The network is contained within dqn.py. We will train our network using run_dqn_pong.py. The environment will keep playing games of pong until either player gets 21 points.

## Setting
For the Q learner we must represent the game as a set of states, actions, and rewards. OpenAI offers the state as the hardware ram for the enviroment for our agent to learn. 

## Making Q-learner learn
Q learner is a type of reinforcement learning that seeks next optimal action with a given current state. The next action is chosen at random and aims to maximize the reward.

In this project, we will use the following Loss function:
	      $$Loss_i(\Theta_i)=(y_i-Q(s,a;\Theta_i))^2$$

This loss function is the "model output" or the square of the q-value minus the q-value in the current state. As time goes on, this value will approach zero, and the closer it is to zero, the better the model is trained. Therefore, it can be used as a method to determine the learning effect of the model. In the Loss function, $\Theta$ corresponds to the model parameters and $y$ corresponds to the model output, while $Q()$ corresponds to the q-value function, which will give the $Q$ value of a given state/action pair, where $s$, a refers to said pair. When the subscripts are added, $\Theta_i$ and $y_i$ refer to the model parameters and output of the i-th iteration respectively. 

In terms of the actual parameters passed to the "compute_td_loss" function in our code, these are model, target_model, batch_size, gamma and replay_buffer. Model is the current environment being trained, and it is used to find the Q-values of the state variables that will be trained and the Q-values of the next_state variables, which will be explained later. target_model is similar, but it is not the training environment, but the last saved environment as the target_model file. batch_size and replay_buffer are complementary, as batch_size simply controls the size of the random samples obtained from replay_buffer, and replay_buffer returns the tensor of variables such as “state", "action", "forward", "next_state" and "done". This is ultimately used to actually calculate the given loss function. Gamma is simply provided gamma, which is used to control whether the Q-learner tries to get a short-term greedy return or a potentially higher long-term return in the end. In our case, we experimented with several different gamma values, but found that 0.99 value was the best performer. Although the loss function is not directly given as an argument, the variables "state", "action", "reward", "next_state" and "done" mentioned earlier are very important. The simplest one is "done", which determines whether the game is over or not, and if it is, the q value should be zero. This is because the next game does not affect the current game


## Learning to Play Pong
To help combat the difficulty in training time, we used a provided network that is partially trained. Moreover, to prevent code from crashing or server from getting shut off, we used torch.save to save our model occasionally when training neural networds.

We also used run_dqn_pong.py currently to record the loss and rewards in losses and all rewards respectively. We continued to train the model by running run_dqn_pong.py. To achieve good performance, we trained the model for approximately 500,000 more frames. We want to optimize different values for hyper-parameters such as γ and the size of the replay buffer.

When we tried training our model the first few times, we noticed that it would start around -20 average reward, go up to around -18, then go back down to -20 after around 120k frames. At this point, we would Ctrl-C and try making changes since we assumed it wasn’t working. One of the best changes we made to prevent it from going back down in reward averages was changing the way that we saved the model. Instead of simply saving it every 5k frames as we did at first, we modified it so that it would save the maximum average reward for the last 10k frames (the same value it prints in the console). Whenever a new maximum average was reached, the model would be saved. After 1 million frames, this achieved an average reward of 19.5. When running test_dqn_pong.py it won every game except for one game. However, after running another training session on that same model, we found the rewards incresaed much faster than the first one.

![loss](https://user-images.githubusercontent.com/118039163/206359690-e2936b17-afa6-4ae1-a6b2-f930f0e472da.jpeg)
![reward](https://user-images.githubusercontent.com/118039163/206359735-973a7476-1337-4d2d-b2f7-37db18259964.jpeg)


## Group Member

Ye Cui - cycui@ucdavis.edu

Yuxin Chen - vyxchen@ucdavis.edu

Yizhang Huang - yzhua@ucdavis.edu
