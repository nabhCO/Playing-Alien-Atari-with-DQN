import torch
import torch.nn as nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

#took a lot of inspo from the PyTorch DQN tutorial for constructing this class
#resource used for calculating convolution output in references
class ConvNeuralNet(nn.Module):

    def __init__(self):

        super(ConvNeuralNet, self).__init__()
        self.conv_layer_1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv_layer_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.dense_layer = nn.Linear(in_features=32 * 9 * 9, out_features=256)
        self.output_layer = nn.Linear(in_features=256, out_features=18)


    #call to pass data through network
    def forward(self, input):

        input = torch.nn.functional.relu(self.conv_layer_1(input))
        print("succeeded layer 1")
        input = torch.nn.functional.relu(self.conv_layer_2(input))
        print("succeeded layer 2")
        input = torch.flatten(input)
        print("succeeded flatten")
        input = torch.nn.functional.relu(self.dense_layer(input))
        print("succeeded layer 3")
        return self.output_layer(input)


#testing with an empty tensor (same dimensions as my actual input will be)
x = torch.empty(4, 84, 84)

model = ConvNeuralNet()
result = model(x)
print(result)
    
        


        





    




        

