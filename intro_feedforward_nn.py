import torch
import torch.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(self):
        '''initialises PINN'''
        super().__init__()
        # define linear hidden layer: 1 input, 10 output
        self.hidden_layer = nn.Linear(1, 10)
        # define nonlinear activation function
        self.activation = nn.Tanh()
        # define final output layer: 10 hidden to 1 output
        self.output_layer = nn.Linear(10, 1)

    def forward(self, x):
        '''executes forward pass'''
        z = self.activation(self.hidden_layer(x))
        return self.output_layer(z)
