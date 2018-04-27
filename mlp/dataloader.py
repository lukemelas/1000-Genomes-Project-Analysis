import pickle
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
use_gpu = torch.cuda.is_available()

def get_dataloader(filepath, batch_size, test_size=0.2, val_size=0.2, seed=[42,74]):
    '''Takes a filepath of standardized (!) PCA components as a numpy array'''

    # Load data
    data = pickle.load(open(filepath, 'rb'))
    print('Loaded data with shape: {}.'.format(data.shape))

    # Load labels (populations)
    pops = pickle.load(open('../data/pops_with_ints_pandas.pkl','rb'))
    labels = pops['pop int'].values
    #spops = pickle.load(open('../data/super_pop_int_numpy.pkl','rb'))
    #labels = spops
    print('Loaded labels with shape: {}.'.format(labels.shape))

    # Test/val split
    X, X_test, y, y_test = train_test_split(data, labels, test_size=test_size, random_state=seed[0])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=seed[1])

    # Torchify
    X_train, X_val, X_test = [torch.FloatTensor(s) for s in [X_train, X_val, X_test]]
    y_train, y_val, y_test = [torch.LongTensor(s) for s in [y_train, y_val, y_test]]
    
    # Constants 
    input_size = X_train.size(1) # N x input_size
    num_classes = y_train.max() + 1 # 0...max

    # Create dataset
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # Create dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, input_size, num_classes

