import torch
import torch.nn as nn
import random
from collections import namedtuple

'''
CLASS DEFINITIONS
'''

'''
ConvNeuralNet(nn.module)

The convolutional neural network (CNN) used as our model. I tried to copy the architecture used in the original Deepmind
paper, with some other resources for the flattening operation. Inputs and outputs were calculated by hand 
using a reference (link to these resources in the references)

Methods:

forward(self, input)
-input (Stack of 4 grayscale images converted to a Tensor (or a batch of these))

Used to pass input to an instance of the model.
'''
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
Experience = namedtuple("Experience", ("state", "action", "reward", "nextState"))

    
'''
ReplayBuffer

An array to store Experience objects. We store these for retrieval during training.

Parameters: 
-capacity (total number of experiences that can be stored at one time in the buffer)

Methods:

push(self, experience)
-input (an Experience object to add to the buffer)

Adds an experience to the buffer.

buffer_length(self)

Returns current number of experiences in the buffer.

random_experience_batch(self)

Randomly sample 32 experiences from the buffer.
'''
class ReplayBuffer:

    def __init__(self, capacity):

        self.replay_buffer = []
        self.buffer_capacity = capacity

    def push(self, experience):
        self.replay_buffer.append(experience)

    def buffer_length(self):
        return len(self.replay_buffer)

    def random_experience_batch(self):
        return random.sample(self.replay_buffer, 32)

    






    




        

