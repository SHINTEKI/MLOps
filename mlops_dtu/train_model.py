import os
folders = [folder for folder in os.listdir() if os.path.isdir(folder)]

# 打印文件夹列表
print("当前目录下的文件夹：")
for folder in folders:
    print(folder)
    
import click
import matplotlib.pyplot as plt
import numpy as np
import torch

# from function import accuracy
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader

from mlops_dtu.models.model import Model


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epoch", default=10, help="epochs for training")
def train(lr, epoch):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here

    # import processed data and make train_loader
    train_x = torch.load("data/processed/normalized_train_x.pt")
    train_y = torch.load("data/processed/normalized_train_y.pt")
    train_dataset = Data.TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # make model
    model = Model()
    # train_loader, _ = corrupted_mnist()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_epochs = epoch
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
        print("epoch:", epoch, " loss:", running_loss)
        train_losses.append(running_loss)
        state_dict.append(model.state_dict())
    x = np.arange(0, num_epochs)
    plt.plot(x, train_losses)
    ind = train_losses.index(min(train_losses))
    op_model = state_dict[ind]

    # save model to pth file
    save_dir = "models/trained"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(op_model, os.path.join(save_dir, "checkpoint.pth"))

    # save the graph to reports
    save_dir = "reports/figures"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "train_loss.png"))


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    state_dict = torch.load(model_checkpoint)
    # _, test_loader = corrupted_mnist()
    model = Model()
    model.load_state_dict(state_dict)
    with torch.no_grad():
        model.eval()
        # acc = 0
        # for batch_idx, (images, labels) in enumerate(get_loader):
        # score = model(images)
        # accuracy_score = accuracy(score, labels)
        # acc += accuracy_score
        # print(acc.item()/(batch_idx+1))


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
