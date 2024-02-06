import torch
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F # All functions that don't have any parameters
from torch.utils.data import DataLoader # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets # Has standard datasets we can import in a nice and easy way
import torchvision.transforms as transforms # Transformations we can perform on our dataset
from torch.nn.parameter import Parameter

import pandas as pd 
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
                
    def forward(self, x): 
        x = x.view(x.shape[0],-1)
        x = self.dropout(self.ReLU(self.fc1(x)))
        x = self.dropout(self.ReLU(self.fc2(x)))
        x = self.dropout(self.ReLU(self.fc3(x)))
        x = self.fc4(x)      
        return x
