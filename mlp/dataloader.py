import pickle
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
use_gpu = torch.cuda.is_available()

def get_dataloader(filepath, batch_size, test_size=0.33, random_seed=42):
    '''Takes a filepath of standardized (!) PCA components as a numpy array'''

    # Load data
    data = pickle.load(open(filepath, 'rb'))
    print('Loaded data with shape: {}.'.format(data.shape))

    # Load labels (populations)
    pops = pickle.load(open('../data/pops_with_ints_pandas.pkl','rb'))
    labels = pops['pop int'].values
    print('Loaded labels with shape: {}.'.format(labels.shape))

    # Test/val split
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.33, random_state=42)

    # Torchify
    X_train, X_val = [torch.FloatTensor(s) for s in [X_train, X_val]]
    y_train, y_val = [torch.LongTensor(s) for s in [y_train, y_val]]
    
    # Constants 
    input_size = X_train.size(1) # N x input_size
    num_classes = y_train.max() + 1 # 0...max

    # Create dataset
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Create dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, input_size, num_classes

