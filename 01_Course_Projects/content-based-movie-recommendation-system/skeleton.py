from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math, random, sys
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from dqn import QLearner, compute_td_loss, ReplayBuffer
from collections import deque
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.cross_decomposition import CCA

class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def getHidden(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        i = 0
        for layer in self.fc:
            if i == 2:
                break
            x = layer(x)
            i += 1
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)        
            q_vals = self(state) # The q-values for each action are the output of the model
            activations = self.getHidden(state).detach().numpy()
            action = q_vals.max(1)[1] # the index of the maximum q-value corresponds to the action we should take
        else:
            action = random.randrange(self.env.action_space.n)
        return action, activations

    def copy_from(self, target):
        self.load_state_dict(target.state_dict())

def applyPCA(X):
    pca = PCA(n_components=2)
    return pca.fit_transform(X)

def applyTSNE(X):
    return TSNE(n_components=2).fit_transform(X)

def applyCCA(X):
    cca=CCA(n_components=2)
    return cca.fit_transform(X)

def applyMDS(X):
    return MDS(n_components=2).fit_transform(X)

MODEL_FILEPATH = 'winning_model.pth'#model_pretrained

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

num_frames = 1000000
batch_size = 32
gamma = 0.99

replay_initial = 10000
replay_buffer = ReplayBuffer(100000)
model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)

model.load_state_dict(torch.load(MODEL_FILEPATH,map_location='cpu'))
model.eval()

info = []
activations = []

env.seed(1)
state = env.reset()
done = False

games_won = 0

while not done:

    action, activ = model.act(state, 0)

    state, reward, done, _ = env.step(action)
    
    info.append((state, action, reward))
    activations.append(activ.reshape((512)))

input(len(activations))
tsne=applyTSNE(activations)
ypca=applyPCA(activations)
ymds=applyMDS(activations)
input(tsne)
input(ypca)
input(ymds)
plt.plot(tsne[:])
plt.show()
plt.plot(ypca[:])
plt.show()

plt.plot(ymds[:])
plt.show()


