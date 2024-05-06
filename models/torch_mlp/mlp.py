import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, viewerdirs):
        x = F.relu(self.input_layer(x))
        x = x[...,:16]
        viewerdirs = viewerdirs[...,:16]
        x = torch.cat((x, viewerdirs), dim=3)
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        x = x[...,:4]
        return x