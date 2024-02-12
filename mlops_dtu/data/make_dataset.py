import os

import torch


def normalized_mnist():
    """Load in the corrupted mnist dataset"""

    # load dataset
    train_images_0 = torch.load("data/raw/corruptmnist/train_images_0.pt")
    train_images_1 = torch.load("data/raw/corruptmnist/train_images_1.pt")
    train_images_2 = torch.load("data/raw/corruptmnist/train_images_2.pt")
    train_images_3 = torch.load("data/raw/corruptmnist/train_images_3.pt")
    train_images_4 = torch.load("data/raw/corruptmnist/train_images_4.pt")
    train_images_5 = torch.load("data/raw/corruptmnist/train_images_5.pt")
    train_target_0 = torch.load("data/raw/corruptmnist/train_target_0.pt")
    train_target_1 = torch.load("data/raw/corruptmnist/train_target_1.pt")
    train_target_2 = torch.load("data/raw/corruptmnist/train_target_2.pt")
    train_target_3 = torch.load("data/raw/corruptmnist/train_target_3.pt")
    train_target_4 = torch.load("data/raw/corruptmnist/train_target_4.pt")
    train_target_5 = torch.load("data/raw/corruptmnist/train_target_5.pt")
    test_x = torch.load("data/raw/corruptmnist/test_images.pt")  # (5000,28,28)
    test_y = torch.load("data/raw/corruptmnist/test_target.pt")

    # concate subsets to form a complete training set for the images and labels
    train_x = torch.cat(
        (train_images_0, train_images_1, train_images_2, train_images_3, train_images_4, train_images_5)
    )  # (30000,28,28)
    train_y = torch.cat(
        (train_target_0, train_target_1, train_target_2, train_target_3, train_target_4, train_target_5)
    )  # (30000,)

    # normalize images in training and test sets to have mean 0 and standard deviation 1
    train_mean = torch.mean(train_x, dim=(1, 2), keepdim=True)
    train_std = torch.std(train_x, dim=(1, 2), keepdim=True)
    test_mean = torch.mean(test_x, dim=(1, 2), keepdim=True)
    test_std = torch.std(test_x, dim=(1, 2), keepdim=True)

    train_x = (train_x - train_mean) / train_std
    test_x = (test_x - test_mean) / test_std

    # zip images and labels to form a complete training set
    # train_dataset = Data.TensorDataset(train_x, train_y)
    # test_dataset = Data.TensorDataset(test_x, test_y)

    # use dataloader to set batch size for each iteration
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)

    # 创建保存标准化后张量的文件夹（如果不存在）
    save_dir = "data/processed"
    os.makedirs(save_dir, exist_ok=True)

    # 保存标准化后的张量
    torch.save(train_x, os.path.join(save_dir, "normalized_train_x.pt"))
    torch.save(train_y, os.path.join(save_dir, "normalized_train_y.pt"))
    torch.save(test_x, os.path.join(save_dir, "normalized_test_x.pt"))
    torch.save(test_y, os.path.join(save_dir, "normalized_test_y.pt"))


if __name__ == "__main__":
    # Get the data and process it
    normalized_mnist()
