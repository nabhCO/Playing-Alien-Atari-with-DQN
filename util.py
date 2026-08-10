import torch
import torchvision

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