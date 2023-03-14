import torch
import numpy as np
#t1 = torch.tensor([[[1],[2]],[[4],[5]]])
#print(t1)
# alwys create a floating point tensor
#print(t1.shape)
# tensor shoyld have a proper shape

#create tensors:

#x = torch.tensor(3.)# we are not intrested in any future outputs wrt x
#w = torch.tensor(4.,requires_grad=True)
#b = torch.tensor(5.,requires_grad=True)
#print(x,w,b)

#arithmatic Operations:
#y = w*x + b
#print(y)

#y.backward()
#print('dy/dx' , x.grad)# prints NONE
#print('dy/dw' , w.grad)
#print('dy/db' , b.grad)

# There are many Tensor operation which needs to be done rn like : full, reshape, cat,sin/cos etc..

# Numpy:

# x = np.array([# a Numpy array
#     [1,2],
#     [3,4.]
#     ])
# # print(x)
# # converting to a tensor
# y = torch.from_numpy(x)
# # print(y) 
# print(x.dtype,y.dtype)

## gradient descent and the Linear Regression:

inputs = np.array([
    [73,67,43],
    [91,88,64],
    [87,134,58],
    [102,43,37],
    [69,96,70],
],dtype='float32')# numpy array because csv to matrix is a method of numpy

targets = np.array([
    [56,70],
    [81,101],
    [119,133],
    [22,37],
    [103,119],
],dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# print(inputs.dtype)
# print(targets.dtype)
# Linear regression mOdel

# initializing the weights and the biases

w = torch.randn(2,3,requires_grad=True)
b = torch.randn(1,2,requires_grad=True)

# print(w,b)

def model(x):
    return x @ w.t() + b

#Generate Predictions: 

# preds = model(inputs)
# print(preds)

# Compare with the targets:

# Here comes the Loss function!!!
# diff = (preds - targets)
# print(diff.numel())
# loss = torch.sum(diff*diff) / diff.numel()
# print(loss)

# Just a function for above task
# def mse(genrated,real):
#     diff = torch.abs(genrated-real)
#     return torch.sum(diff*diff) / diff.numel()

## compute the loss:
# loss = mse(preds,targets)
# print('loss=',loss)

## Compute the Gradients:
# loss.backward()

## Gradients For weights:
# print(w)
# print(w.grad)# derivative of the loss wrt to the w.
# print(b.grad)# derivative of the loss wrt to the b.
# lr = 1e-5
## Adjust weights and biases to reducd the loss

# with torch.no_grad():
#     w -= w.grad*lr
#     b -= b.grad*lr

def mse(pred,targets):
    diff = torch.abs(pred-targets)
    return torch.sum(diff*diff)/diff.numel()

EPOCH = 200
lr  = 1e-5
def train():
    global w, b
    for epoch in range(EPOCH) :
        #pridiction:
        pred = model(inputs)
        #loss:
        loss = mse(pred,targets)
        print('epoch: ',epoch,' loss:',loss)
        #Update the params:
        loss.backward()
        with torch.no_grad():
            w -= w.grad*lr
            b -= b.grad*lr
            w.grad.zero_()
            b.grad.zero_()


train()

fin = model(inputs)
print(fin)








    


