import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions


class Model(nn.Module):
    """Basic neural network class.

    Args:
        in_features: number of input features
        out_features: number of output features

    """

    def __init__(self, in_features=784, out_features=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, out_features)
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        x = x.view(x.shape[0], -1)
        x = self.dropout(self.ReLU(self.fc1(x)))
        x = self.dropout(self.ReLU(self.fc2(x)))
        x = self.dropout(self.ReLU(self.fc3(x)))
        x = self.fc4(x)
        return x
