import os, sys, time, datetime
import pandas as pd
import numpy as np
import torch
import sklearn
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Load PCA with n=1000
data_new = pickle.load(open('data/data_pca_300comps.pkl','rb'))
print('Data with n=1000 shape: ', data_new.shape)

# Load labels (super-populations and populations)
spop_ints = pickle.load(open('data/super_pop_int_numpy.pkl','rb'))
pop_ints = pickle.load(open('data/pops_with_ints_pandas.pkl','rb'))
pop_ints = pop_ints['pop int'].values
print('Super pop labels shape: ', spop_ints.shape, spop_ints) # y
print('Pop labels shape: ', pop_ints.shape, pop_ints) # y

# Naming
X = data_new
y = pop_ints

# Models
lr = LogisticRegression()
rf = RandomForestClassifier(n_estimators=100)
gb = GradientBoostingClassifier(n_estimators=100)

# Cross validation
accuracy_lr = cross_val_score(lr, X, y, scoring='accuracy', cv = 5)
print('LR Accuracy: {:.2f}'.format(accuracy_lr.mean() * 100))

accuracy_rf = cross_val_score(rf, X, y, scoring='accuracy', cv = 5)
print('RF Accuracy: {:.2f}'.format(accuracy_rf.mean() * 100))

accuracy_gb = cross_val_score(gb, X, y, scoring='accuracy', cv = 5)
print('GB Accuracy: {:.2f}'.format(accuracy_gb.mean() * 100))

# Save results
torch.save({'LR': accuracy_lr, 'RF': accuracy_rf, 'GB': accuracy_gb}, 'results_baselines_300comps.pth')

