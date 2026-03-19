import torch
import torch.nn as nn

class CVDModel(nn.Module):
    def __init__(self, input_size, list_hidden, activation='relu', dropout=0.2):
        super().__init__()
        self.activation_type = activation
        
        layers = []
        curr_dim = input_size
        
        # Build Hidden Layers
        for h in list_hidden:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(self.get_activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_dim = h
            
        self.main = nn.Sequential(*layers)
        self.output = nn.Linear(curr_dim, 3) # 3 Classes: Low, Inter, High
        
        self.init_weights()

    def get_activation(self):
        if self.activation_type == 'tanh':
            return nn.Tanh()
        elif self.activation_type == 'sigmoid':
            return nn.Sigmoid()
        elif self.activation_type == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            return nn.ReLU()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight) # Better for ReLU
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        return self.output(self.main(x))