import os, sys, time, datetime
import pickle

import pandas as pd
import numpy as np
import torch
import sklearn

from sklearn.decomposition import IncrementalPCA

import pdb

data_float = pickle.load(open('data/data_scaled_float16.pkl', 'rb')) # a standardized numpy float16 array
print('Data loaded.') 

ipca = IncrementalPCA(n_components=1000, batch_size=2000)
ipca.fit(data_float)
print('IncrementalPCA fitted.')

data_new = ipca.transform(data_float)
print('PCA applied to data.')

pickle.dump(data_new, open('data_pca_1000comps.pkl', 'wb'))
print('Saved post-PCA data')

