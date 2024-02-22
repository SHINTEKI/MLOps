import click

# from function import accuracy
import numpy as np
import torch

from mlops_dtu.models.model import Model


@click.group()
def cli():
    pass


@click.command()
@click.option("--model", default="models/trained/checkpoint.pth", help="model used for prediction")
@click.option("--data", default="data/raw/example_data.npz", help="data used for prediciton")
def predict(model, data):
    # Preprocess the image
    data_x = torch.tensor(np.load(data)["images"]).float()
    x_mean = torch.mean(data_x, dim=(1, 2), keepdim=True)
    x_std = torch.std(data_x, dim=(1, 2), keepdim=True)
    data_x = (data_x - x_mean) / x_std

    # Use the model to predict the label
    results = []
    model_pred = Model()
    state_dict = torch.load(model)
    model_pred.load_state_dict(state_dict)
    model_pred.eval()
    with torch.no_grad():
        outputs = model_pred(data_x)
        _, predicted = torch.max(outputs, 1)
        results.extend(predicted.numpy())
        print(predicted)
    return results


# Add the predict command to the CLI
cli.add_command(predict)

if __name__ == "__main__":
    cli()
