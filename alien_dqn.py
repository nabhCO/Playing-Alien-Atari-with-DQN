import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, FrameStackObservation
from ale_py import ALEInterface

ale = ALEInterface()

#setting up environment and preprocessing- observation are image stacks of 4 that are grayscale, 84x84
env = gym.make('ALE/Alien-v5', obs_type="grayscale", render_mode="human") #change render_mode while training
env = ResizeObservation(env, 84)
env = FrameStackObservation(env, 4) # return stack of most recent 4 observations


