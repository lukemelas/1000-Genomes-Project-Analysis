import torch.nn as nn

import pdb

class ExperimentalModel(nn.Module):
    def __init__(self, input_size, num_classes, dropout):
        super(ExperimentalModel, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        
        self.linear4 = nn.Linear(3, 1)
        self.linear5 = nn.Linear(self.input_size, self.num_classes)

        #self.linear1 = nn.Linear(self.input_size, 100)
        #self.linear2 = nn.Linear(50, 50)
        #self.linear3 = nn.Linear(100, self.num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear4(x)
        x = x.squeeze()
        x = self.relu(x)
        x = self.dropout(x) 
        x = self.linear5(x)
        #x = self.relu(self.linear1(x))
        #x = self.dropout(x)
        #x = self.relu(x + self.linear2(x))
        #x = self.dropout(x)
        #x = self.linear3(x)
        return x
