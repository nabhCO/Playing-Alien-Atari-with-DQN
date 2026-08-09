import gymnasium as gym
import torch
import torchvision
import numpy as np
from util import transform_obs
from parameters import gamma, alpha, epsilon, epsilon_decay, batch_size, num_episodes
from CNN_DQNAgent import ConvNeuralNet, ReplayBuffer
from gymnasium.wrappers import ResizeObservation, FrameStackObservation
from ale_py import ALEInterface


ale = ALEInterface()

#setting up environment and preprocessing- observation are image stacks of 4 that are grayscale, 84x84
env = gym.make('ALE/Alien-v5', obs_type="grayscale", render_mode="human") #change render_mode while training
env = ResizeObservation(env, (84, 84))
env = FrameStackObservation(env, 4) # return stack of most recent 4 observations

#initialize model, optimizer, replay buffer
model = ConvNeuralNet()
optimizer = torch.optim.RMSprop(model.parameters(), lr=alpha)
replay_buffer = ReplayBuffer(500)

#function to select action based on epsilon greedy strategy, decay epsilon value
def choose_action(obs, epsilon):

    choose_random_action = np.random.choice(a=[True, False], p=[epsilon, 1-epsilon])

    #apply linear epsilon decay as we will take a step with the action we choose
    new_epsilon = epsilon * epsilon_decay

    #with probability epsilon, we choose a random action to explore
    if choose_random_action:

        action = env.action_space.sample()

        #we're putting action in this form so it matches the model output (with an extra dimension for batch size)
        return torch.tensor([[action]], dtype=torch.long), new_epsilon

    #with probability 1 - epsilon, we use our model to decide the action for us
    #don't calculate gradients, we're not training the model right now
    with torch.no_grad():

        obs = transform_obs(obs)
        action_vals = model.forward(obs)
        return action_vals.max(1).indices.view(1, 1), new_epsilon #returns the index for max value action

    
    



    




