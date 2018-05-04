import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout):
        super(MLP, self).__init__()
        self.input_size     = input_size
        self.hidden_size    = hidden_size
        self.num_classes    = num_classes
        self.dropout        = dropout

        self.module1 = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.module2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.module3 = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_classes)
        )

    def forward(self, x):
        x = self.module1(x)
        x = self.module3(x)
        return x
