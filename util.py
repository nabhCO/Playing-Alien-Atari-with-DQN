import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

'''
transform_obs(obs)
Parameters:
-obs (a stack of 4 grayscale images)

Description:
Converts a stack of 4 grayscale images to a Tensor object with dimension (batch size, width, height, channels). This is 
suitable input for our model.

The torchvision library is used here because I was having some trouble with dtype from the usual transform function. Some
StackOverflow post responses suggested using this library for anything vision related instead.
'''
def transform_obs(obs):

    transform = torchvision.transforms.ToTensor()
    obs = transform(obs)
    obs = torch.permute(obs, (1, 2, 0))
    return obs.unsqueeze(0) #add a dimension at index 0 (this is for batch size)


def plot_avg_reward(avg_reward, epochs, num_episodes):

    x_axis = np.array(epochs)
    y_axis = np.array(avg_reward)

    plt.xlim(0, num_episodes / 10)
    plt.xticks(np.arange(0, (num_episodes/10) + 1, 10))

    plt.xlabel("Epochs (1 Epoch = 10 Episodes)")
    plt.ylabel("Average Reward")
    plt.title(f"Average Rewards Over {num_episodes} Epochs")

    plt.plot(x_axis, y_axis)
    plt.savefig("avg_reward_over_time.jpg")

