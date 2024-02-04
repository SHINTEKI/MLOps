import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms 
from torch.nn.parameter import Parameter
import torch.utils.data as Data

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


def corrupted_mnist():
    """Load in the corrupted mnist dataset"""
    
    # load dataset
    train_0 = np.load("F:/Master's/DTU courses/MLOps/dtu_mlops/data/corruptmnist/train_0.npz",allow_pickle=True)
    train_1 = np.load("F:/Master's/DTU Courses/MLOps/dtu_mlops/data/corruptmnist/train_1.npz",allow_pickle=True)
    train_2 = np.load("F:/Master's/DTU Courses/MLOps/dtu_mlops/data/corruptmnist/train_2.npz",allow_pickle=True)
    train_3 = np.load("F:/Master's/DTU Courses/MLOps/dtu_mlops/data/corruptmnist/train_3.npz",allow_pickle=True)
    train_4 = np.load("F:/Master's/DTU Courses/MLOps/dtu_mlops/data/corruptmnist/train_4.npz",allow_pickle=True)
    test = np.load("F:/Master's/DTU Courses/MLOps/dtu_mlops/data/corruptmnist/test.npz",allow_pickle=True)
    
    # transform numpy array to tensor
    train_0_x = torch.from_numpy(train_0["images"])
    train_1_x = torch.from_numpy(train_1["images"])
    train_2_x = torch.from_numpy(train_2["images"])
    train_3_x = torch.from_numpy(train_3["images"])
    train_4_x = torch.from_numpy(train_4["images"])
    train_0_y = torch.from_numpy(train_0["labels"])
    train_1_y = torch.from_numpy(train_1["labels"])
    train_2_y = torch.from_numpy(train_2["labels"])
    train_3_y = torch.from_numpy(train_3["labels"])
    train_4_y = torch.from_numpy(train_4["labels"])
    test_x = torch.from_numpy(test["images"])
    test_y = torch.from_numpy(test["labels"])
    
    # concate subsets to form a complete training set for the images and labels 
    train_x = torch.cat((train_0_x, train_1_x, train_2_x, train_3_x, train_4_x)) # (25000,28,28)
    train_y = torch.cat((train_0_y, train_1_y, train_2_y, train_3_y, train_4_y)) # (25000,)
    
    # normalize images in training set to fall in the range of [-1,1] 
    train_x = ((train_x - 0.5)/0.5).to(torch.float32)
    test_x = ((test_x - 0.5)/0.5).to(torch.float32)

    # zip images and labels to form a complete training set
    train_dataset = Data.TensorDataset(train_x, train_y)
    test_dataset = Data.TensorDataset(test_x, test_y)
    
    # use dataloader to set batch size for each iteration
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)
    
    return train_loader, test_loader
