import torch.nn as nn

import pdb

class ExperimentalModel(nn.Module):
    def __init__(self, input_size, pca_size, hidden_size, num_classes, dropout):
        super(ExperimentalModel, self).__init__()
        self.input_size     = input_size
        self.pca_size       = pca_size
        self.hidden_size    = hidden_size
        self.num_classes    = num_classes
        self.dropout        = dropout

        self.first_layer = nn.Linear(self.input_size, self.pca_size)

        self.module1 = nn.Sequential(
            nn.Linear(self.pca_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.module2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.module3 = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_classes)
        )

    def forward(self, x):
        x = self.first_layer(x)
        x = self.module1(x)
        x = self.module3(x)
        return x
    
