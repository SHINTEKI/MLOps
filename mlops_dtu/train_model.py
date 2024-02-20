import os  
import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

# from function import accuracy
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader

from mlops_dtu.models.model import Model

wandb.init(project="MLOps", entity="wuxindi1127")

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
    wandb.watch(model, log_freq=100)
    # train_loader, _ = corrupted_mnist()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_epochs = epoch
    train_losses = []
    state_dict = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
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
            if batch_idx % 100 == 0:
                wandb.log({"loss": loss.item()})

            _, pred = torch.max(scores.data, 1)
        print("epoch:", epoch, " loss:", running_loss)
        train_losses.append(running_loss)
        state_dict.append(model.state_dict())
    
    # log 
    # wandb.log({"examples": [wandb.Image(im) for im in images]})
    
    my_table = wandb.Table(columns=["image_id", "image", "label", "prediction"])
    
    for img_id, img in enumerate(images):
        true_label = labels[img_id]
        guess_label = pred[img_id]
        my_table.add_data(img_id, wandb.Image(img),true_label, guess_label)
    wandb.log({"mnist_predictions": my_table})

    # my_table.add_column("image", wandb.Image(images.tolist()))
    # my_table.add_column("label", labels.tolist())
    # my_table.add_column("class_prediction", pred.tolist())
    # wandb.log({"mnist_predictions": my_table})
    
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
