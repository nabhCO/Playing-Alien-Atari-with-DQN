'''
HYPERPARAMETERS
'''
gamma = 0.4 # 1 for now, no prioritization of closer rewards at the moment
alpha = 0.1 # for backprop, 0.1, 0.01, 0.001 are options- NEVER 1
epsilon = 0.9 # decay as we get more experiences to learn from (in range 0.1, 0.9 inclusive)
epsilon_decay = 0.4 # decay value
batch_size = 32 # number of experiences to be taken from replay buffer during training
num_episodes = 100 # 1 for now because we just want to see if the model's going to run for an episode
