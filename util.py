import torch
import torchvision

# takes an gym environment observation (game frames in this case) and converts to Tensor object with dimension (batch size, width, height, channels)
def transform_obs(obs):

    transform = torchvision.transforms.ToTensor()
    obs = transform(obs)
    obs = torch.permute(obs, (1, 2, 0))
    return obs.unsqueeze(0) #add a dimension at index 0 (this is for batch size)



