from mlops_dtu.models.model import Model
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import click

@click.group()
def cli():
    pass


@click.command()
@click.option("--model", default="models/trained/checkpoint.pth", help='model used for prediction')
@click.option("--data", default="data/raw/example_data.npz", help='data used for prediciton')
def plot(model, data):
    """
    Loads a pre-trained network
    Extracts some intermediate representation of the data (your training set) from your cnn. This could be the features just before the final classification layer
    Visualize features in a 2D space using t-SNE to do the dimensionality reduction.
    Save the visualization to a file in the reports/figures/ folder
    """
    # Preprocess the image
    data_x = torch.tensor(np.load(data)["images"]).float()[:500]
    x_mean = torch.mean(data_x, dim=(1,2), keepdim=True)
    x_std = torch.std(data_x, dim=(1,2), keepdim=True)  
    data_x = (data_x - x_mean)/x_std
    

    # load a pre-trained model
    model_pred = Model()
    state_dict = torch.load(model)
    model_pred.load_state_dict(state_dict)

    # extract intermediate representation of the training set/ features just before the classification layer    
    results = []
    model_pred.eval()
    with torch.no_grad():
        outputs = model_pred(data_x)
        _, predicted = torch.max(outputs, 1)
        results.extend(predicted.numpy())
        results_array = np.hstack(results)
    
    # t-sne
    outputs_std = StandardScaler().fit_transform(outputs)
    tsne = TSNE(n_components=2, init="random") 
    outputs_tsne = tsne.fit_transform(outputs_std) 
    X_tsne_data = np.vstack((outputs_tsne.T, results_array)).T 
    df_tsne = pd.DataFrame(X_tsne_data, columns=["Dim1", "Dim2", "class"])  

    plt.figure(figsize=(8, 8)) 
    sns.scatterplot(data=df_tsne, palette="tab10", hue="class", x="Dim1", y="Dim2") 

    # save the graph to reports
    save_dir = "reports/figures"
    os.makedirs(save_dir, exist_ok=True)  
    plt.savefig(os.path.join(save_dir, 'tsne.png'))
    plt.show()

# Add the predict command to the CLI
cli.add_command(plot)

if __name__ == "__main__":
    cli()