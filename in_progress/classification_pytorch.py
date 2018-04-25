'''WORK IN PROGRESS'''

import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import pdb

# CUDA
use_gpu = torch.cuda.is_available()
gpu_str = 'Using GPU.' if use_gpu else 'Not using GPU'
print(gpu_str)

# Load PCA with n=all
data_new_all = pickle.load(open('data/data_scaled_float16.pkl','rb'))
print('Data with n=all shape: ', data_new_all.shape)

# Load PCA with n=1000
data_new = pickle.load(open('data/data_pca_1000comps.pkl','rb'))
print('Data with n=1000 shape: ', data_new.shape)

# Load labels (populations)
pop_ints = pickle.load(open('data/pops_with_ints_pandas.pkl','rb'))
pop_ints = pop_ints['pop int'].values
print('Pop labels shape: ', pop_ints.shape, pop_ints) # y

# Naming
X = data_new_all
y = pop_ints

# Test/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

# Torchify
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.LongTensor(y_val)

# Create dataset
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

# Validation
def validate(model, val_loader, criterion):
    total = 0
    total_correct = 0
    total_loss = 0
    for input, label in val_loader:
        input = V(input, volatile=True).cuda() if use_gpu else V(input, volatile=True)
        label = V(label, volatile=True).cuda() if use_gpu else V(label, volatile=True)
        output = model(input)
        _, pred = torch.max(output, 1)
        total_correct += (pred == label).long().data.sum()
        loss = criterion(output, label)
        total_loss += loss.data[0]
        total += label.numel()
    return total_correct, total

# Training
def train(model, train_loader, val_loader, optimizer, criterion, num_epochs=30, print_freq=50):
    accs = []
    best_acc = 0
    for epoch in range(num_epochs):

        # One epoch of training
        total = 0
        total_loss = 0
        for i, (input, label) in enumerate(train_loader):
            input = V(input).cuda() if use_gpu else V(input)
            label = V(label).cuda() if use_gpu else V(label)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.data[0]
            total += label.numel()

            if i % print_freq == 0:
                print('Epoch[{}/{}] \t Iter [{:>3}/{:>3}] \t Loss: {:.3f}'.format(
                    epoch + 1, num_epochs, (i+1), len(train_loader), total_loss/total))

        # One epoch of validation
        total_correct, total = validate(model, val_loader, criterion)
        acc = total_correct / total
        accs.append(acc)
        print('Epoch[{}/{}] \t Validation Accuracy {}/{} = {:.3f}% \n'.format(
            epoch + 1, num_epochs, total_correct, total, 100 * acc))

        # Save best model
        if epoch % 20 == 1 and acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'save/model_best.pkl')

    return accs

# Constants
INPUT_SIZE = X_train.size(1)
NUM_CLASSES = y_train.max() + 1 # 0...max
PRINT_FREQ = 100
NUM_EPOCHS = 1000

# Hyperparameters
batch_size = 16
learning_rate = 1e-4
param_init = 0.2

# Create dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create model, optimizer, and loss
model = LogisticRegression(INPUT_SIZE, NUM_CLASSES)
criterion = nn.CrossEntropyLoss(size_average=False)  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Initialize model
for p in model.parameters():
    p.data.uniform_(-param_init, param_init)
if use_gpu:
    model.cuda()

# Train
accs = train(model, train_loader, val_loader, optimizer, criterion, 
        print_freq=PRINT_FREQ, num_epochs=NUM_EPOCHS)

# Print accuracies over epochs
torch.save(accs, 'save/accs.pkl')
