import torch
import os
import click

@click.command()
@click.option('--folder', '-f', default="corruptmnist", help='Path to the folder containing files')
def normalized_mnist(folder):
    """Load in the dataset"""

    base_directory = "data/raw/"
    route = os.path.join(base_directory, folder)

    train_image_files = [file for file in os.listdir(route) if 'train' in file and 'images' in file]
    train_target_files = [file for file in os.listdir(route) if 'train' in file and 'target' in file]
    assert train_image_files 
    assert train_target_files

    # Initialize empty lists to store loaded data
    train_images = []
    train_targets = []

    # Load images and targets
    for image_file, target_file in zip(train_image_files, train_target_files):
        train_images.append(torch.load(os.path.join(route, image_file)))
        train_targets.append(torch.load(os.path.join(route, target_file)))

    # Concatenate data
    train_x = torch.cat(train_images)  # ex, (30000, 28, 28)
    train_y = torch.cat(train_targets)  # (30000,)

    # normalize images in training and test sets to have mean 0 and standard deviation 1
    train_mean = torch.mean(train_x, dim=(1, 2), keepdim=True)
    train_std = torch.std(train_x, dim=(1, 2), keepdim=True)
    # test_mean = torch.mean(test_x, dim=(1, 2), keepdim=True)
    # test_std = torch.std(test_x, dim=(1, 2), keepdim=True)

    train_x = (train_x - train_mean) / train_std
    # test_x = (test_x - test_mean) / test_std

    # zip images and labels to form a complete training set
    # train_dataset = Data.TensorDataset(train_x, train_y)
    # test_dataset = Data.TensorDataset(test_x, test_y)

    # use dataloader to set batch size for each iteration
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)

    # 创建保存标准化后张量的文件夹（如果不存在）
    save_dir = os.path.join("data/processed", folder)   
    os.makedirs(save_dir, exist_ok=True)

    # 保存标准化后的张量
    torch.save(train_x, os.path.join(save_dir, "normalized_train_x.pt"))
    torch.save(train_y, os.path.join(save_dir, "normalized_train_y.pt"))
    # torch.save(test_x, os.path.join(save_dir, "normalized_test_x.pt"))
    # torch.save(test_y, os.path.join(save_dir, "normalized_test_y.pt"))

if __name__ == "__main__":
    # Get the data and process it
    normalized_mnist()
