# implementing the whole pytorch thing using the toch.nn module
import torch
import torch.nn as nn
import numpy as np

# Inputs (temp , rainfall, humidity)
inputs = np.array([
    [ 68,  99,  53],
    [ 89,  66,  98],
    [ 52,  56, 119],
    [102,  87,  76],
    [105, 106,  93],
    [ 55, 102, 101],
    [ 64,  55,  80],
    [ 73,  78,  77],
    [ 78,  59, 115],
    [ 56,  89,  67],
    [ 84,  79,  91],
    [ 55, 115,  75],
    [ 89,  90,  87],
    [113,  77,  97],
    [ 96,  55,  86],
],dtype='float32')

# Targets (apples and oranges)!!
targets = np.array([
    [ 86,  59],
    [102,  61],
    [ 98,  55],
    [ 95,  52],
    [119,  85],
    [ 59,  85],
    [ 87,  73],
    [ 57,  80],
    [106,  57],
    [ 68,  66],
    [ 69, 118],
    [102,  88],
    [114,  83],
    [ 80,  58],
    [ 66, 100],
],dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

## Dataset and Dataloader:

from torch.utils.data import TensorDataset
# define a dataset
train_ds = TensorDataset(inputs,targets)
# This allows us to rows of input and target as tuples
# print(train_ds[0:3])
# basically it lets you access some slice of a data

from torch.utils.data import DataLoader
# Define a dataloader:
# will solit our data into batches

batch_size = 5 # perbatch 5 data points, generally about 100 rakhenge
train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True)

# for xb,yb in train_dl:
#     print(xb)
#     print(yb,'\n')


## Model :

# Defining the model: 

model = nn.Linear(3,3), # 3 inputs and 2 outputs
# print(model.weight)
# print(model.bias)
# Both the parameters have set weights and biases to true.

# print(list(model.parameters()))# This method lists all the parameters contained in our model

# Generating the predictions:
# preds = model(inputs)
# print(preds)

# Define the Loss function:
loss_fn = nn.MSELoss()

# loss = loss_fn(model(inputs),targets)
# print(loss)

## Optimizers:
lr = 1.5e-3
# define the optimizer:
opt = torch.optim.SGD(model.parameters(),lr=lr)

## Train the model:
def train(num_epochs,model,loss_fn,opt,train_dl):
    for epoch in range(num_epochs):
        for xb,yb in train_dl:
            # Generate the predictions:
            pred = model(xb)
            # Calculate the loss:
            loss = loss_fn(pred,yb)
            # Calculate the gradients:
            loss.backward()
            # Update the parms:
            opt.step()
            # resret the gradients to zero:
            opt.zero_grad()
    # Print the progress:
        if((epoch+1)%10==0):
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1,num_epochs,loss.item()))

train(100,model,loss_fn,opt,train_dl)

print(model(inputs),'\n')
print(targets)




