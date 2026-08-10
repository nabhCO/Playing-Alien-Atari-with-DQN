'''
HYPERPARAMETERS
'''

gamma = 1 # 1 for now, no prioritization of closer rewards at the moment
alpha = 0.1 # for backprop, 0.1, 0.01, 0.001 are options- NEVER 1
epsilon = 0.9 # decay as we get more experiences to learn from?
epsilon_decay = 0.0001 # for linear decay
batch_size = 32 # number of experiences to be taken from replay buffer during training
num_episodes = 2 # 1 for now because we just want to see if the model's going to run for an episode
