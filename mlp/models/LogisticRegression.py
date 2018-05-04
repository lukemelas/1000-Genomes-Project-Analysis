import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_size, pca_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(pca_size, num_classes)

    def forward(self, x):
        x = self.linear(x)
        return x


