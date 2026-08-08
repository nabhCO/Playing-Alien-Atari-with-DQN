import gymnasium as gym
import torch
from gymnasium.wrappers import ResizeObservation, FrameStackObservation
from ale_py import ALEInterface


ale = ALEInterface()

#setting up environment and preprocessing- observation are image stacks of 4 that are grayscale, 84x84
env = gym.make('ALE/Alien-v5', obs_type="grayscale", render_mode="human") #change render_mode while training
env = ResizeObservation(env, (84, 84))
env = FrameStackObservation(env, 4) # return stack of most recent 4 observations

#after preprocessing, obs if a numpy.ndarray with dimensions (4, 84, 84)
obs, info = env.reset()

#converting to a tensor
obs = torch.from_numpy(obs)
print(type(obs))

#permute dimensions to match what model expects (width, height, channels) 
obs = torch.permute(obs, (1, 2, 0))
print(obs.shape)


