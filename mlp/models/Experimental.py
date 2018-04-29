import torch.nn as nn

import pdb

class ExperimentalModel(nn.Module):
    def __init__(self, input_size, num_classes, emb_size, dropout):
        super(ExperimentalModel, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.emb_size = emb_size
        
        self.embedding = nn.Linear(self.input_size, self.emb_size)
        self.linear1 = nn.Linear(self.emb_size, 100)
        self.linear2 = nn.Linear(100, self.num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x) 
        x = self.linear2(x)
        return x
