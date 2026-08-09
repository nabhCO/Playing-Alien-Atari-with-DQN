import torch
import torch.nn as nn
import random
import numpy as np
from collections import namedtuple

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

#took a lot of inspo from the PyTorch DQN tutorial for constructing these classes

#architecture for the CNN we'll be training
class ConvNeuralNet(nn.Module):

    def __init__(self):

        #layers and their parameters are from the Deepmind "Playing Atari with DQN" paper
        #resource used for calculating convolution output in references
        super(ConvNeuralNet, self).__init__()
        self.conv_layer_1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv_layer_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.dense_layer = nn.Linear(in_features=32 * 9 * 9, out_features=256)
        self.output_layer = nn.Linear(in_features=256, out_features=18) #18 outputs for 18 available actions


    #call to pass data through network (in this case, a stack of 4 game frames converted to a tensor object)
    def forward(self, input):

        input = torch.nn.functional.relu(self.conv_layer_1(input))
        input = torch.nn.functional.relu(self.conv_layer_2(input))
        input = torch.flatten(input, start_dim=1) #flatten before linear layers
        input = torch.nn.functional.relu(self.dense_layer(input))
        return self.output_layer(input)
    

#named tuple for easier handling of batch tuples and their elements during training
Experience = namedtuple("Experience", ("state", "action", "reward", "nextState", "complete"))

    
#a replay buffer to draw experiences from during training. experience structure is (state, action, reward, nextState, complete)
class ReplayBuffer:

    def __init__(self, capacity):

        self.replay_buffer = []
        self.buffer_capacity = capacity

    #add an experience to the buffer
    def push(self, experience):

        self.replay_buffer.append(experience)

    #check if the buffer's full
    def buffer_at_capacity(self):

        if len(self.replay_buffer) < self.buffer_capacity:
            return False

    #select a random experience from the buffer for training
    def random_experience_batch(self):

        return random.sample(self.replay_buffer, 32)

    


        





    




        

