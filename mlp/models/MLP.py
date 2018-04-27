import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, num_classes, dropout):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = 100
        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
