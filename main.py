import argparse
import sys

import torch
import click
import matplotlib.pyplot as plt  

from data import corrupted_mnist
from model import Model
from function import accuracy

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

@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = Model()
    train_loader, _ = corrupted_mnist()
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_epochs = 10
    train_losses = []
    state_dict = []
    
    for epoch in range(num_epochs):
        
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            
            # forward propagation
            scores = model(images)
            loss = criterion(scores, labels)
            
            # zero previous gradients
            optimizer.zero_grad() 
            
            # back-propagation
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
            
            running_loss += loss.item()
        print("epoch:",epoch," loss:",running_loss)
        train_losses.append(running_loss)
        state_dict.append(model.state_dict())
    x=np.arange(0,num_epochs)
    plt.plot(x,train_losses);
    ind = train_losses.index(min(train_losses))
    op_model = state_dict[ind]
    # save model to pth file
    torch.save(op_model, 'checkpoint.pth')
    

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    state_dict = torch.load(model_checkpoint)
    _, test_loader = corrupted_mnist()
    model = Model()
    model.load_state_dict(state_dict)
    with torch.no_grad():
        model.eval()
        acc = 0
        for batch_idx, (images, labels) in enumerate(test_loader):
            score = model(images)
            accuracy_score = accuracy(score, labels)
            acc += accuracy_score
        print(acc.item()/(batch_idx+1)) 



cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    