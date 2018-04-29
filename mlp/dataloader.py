import pickle
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.utils.data import TensorDataset, DataLoader
use_gpu = torch.cuda.is_available()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

import pdb

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

def _standardize(X_train, X_val, X_test):
    '''Splits data into train/val splits and standardizes'''
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train, copy=False)
    X_val = scaler.transform(X_val, copy=False)
    X_test = scaler.transform(X_test, copy=False)
    return X_train, X_val, X_test

def get_dataloader(preloaded_splits, X, X_test, y, y_test, batch_size=64, val_fraction=0.1, 
        pca_components=200, apply_pca_transform=True, 
        imputation_dim=-1, split=0, save_dataset=True):
    '''Loads pretrained splits or creates new splits. Returns dataloader.'''

    if preloaded_splits.lower() != 'none': 
        # Load preloaded split
        dataset_path = os.path.join(preloaded_splits, 'datasets_{}.pth'.format(split))
        X_train, X_val, X_test, y_train, y_val, y_test, pca_matrix = torch.load(dataset_path)
    else: 
        # Generate split
        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=val_fraction, shuffle=True)
        # Standardize 
        X_train, X_val, X_test = _standardize(X_train, X_val, X_test)
        # PCA
        ipca = IncrementalPCA(n_components=pca_components, batch_size=10*pca_components)
        ipca.fit(X_train)
        pca_matrix = ipca.components_.astype(np.float16)
        # Save dataset
        if save_dataset:
            X_train, X_val, X_test = [x.astype(np.float16, copy=False) for x in [X_train, X_val, X_test]]
            torch.save([X_train, X_val, X_test, y_train, y_val, y_test, pca_matrix], 'save/datasets_{}.pth'.format(split))
            #torch.save(pca_matrix, 'save/pca_matrix_{}.pth'.format(split))

    # Apply PCA for faster training 
    if apply_pca_transform:
        X_train = ipca.transform(X_train)
        X_val = ipca.transform(X_val)
        X_test = ipca.transform(X_test)

    # Torchify
    X_train, X_val, X_test = [torch.FloatTensor(s) for s in [X_train, X_val, X_test]]
    y_train, y_val, y_test = [torch.LongTensor(s) for s in [y_train, y_val, y_test]]
    
    # Constants 
    input_size, num_classes = X_train.size(1), y_train.max() + 1

    # Create dataset
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # Create dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, pca_components, input_size, num_classes

