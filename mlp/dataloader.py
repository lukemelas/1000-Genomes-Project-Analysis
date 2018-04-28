import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.utils.data import TensorDataset, DataLoader
use_gpu = torch.cuda.is_available()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

def get_data(data_path, labels_path):
    '''Loads data and returns numpy array'''

    # Load data
    data = pickle.load(open(data_path, 'rb'))
    data = data.astype(np.float32, copy=False)
    print('Loaded data with shape: {}.'.format(data.shape))

    # Load labels (populations)
    pops = pickle.load(open(labels_path, 'rb')) 
    labels = pops['pop int'].values
    print('Loaded labels with shape: {}.'.format(labels.shape))

    return data, labels

def get_dataloader(data_X, data_X_test, y, y_test, batch_size=64, val_fraction=0.2, pca_components=200):
    '''Performs PCA and returns dataloaders'''

    # Train / val split
    data_X_train, data_X_val, y_train, y_val = train_test_split(data_X, y, test_size=val_fraction)
    
    # Standardization
    scaler = StandardScaler()
    scaler.fit(data_X_train)
    data_scaled_X_train = scaler.transform(data_X_train, copy=False)
    data_scaled_X_val = scaler.transform(data_X_val, copy=False)
    data_scaled_X_test = scaler.transform(data_X_test, copy=False)
    print('Data scaled.')

    # PCA
    ipca = IncrementalPCA(n_components=pca_components, batch_size=10*pca_components)
    ipca.fit(data_scaled_X_train)
    X_train = ipca.transform(data_scaled_X_train)
    X_val = ipca.transform(data_scaled_X_val)
    X_test = ipca.transform(data_scaled_X_test)
    print('PCA performed.')

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

