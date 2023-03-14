## Working with Images:
import torch
import torchvision
import torchvision.datasets as torch_dataset
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader

dataset = torch_dataset.MNIST(root='data/')# Downloads here only!!
# 60,000 data samples:
# print(len(dataset))

test_dataset = torch_dataset.MNIST(root='data/',train=False)
# 10,000 test samples:
# print(len(test_dataset))

# print(dataset[0])
# A pair of Image as a py object and the label.

# Let's Plot the image from teh dataset:


# image ,label = dataset[3]
# plt.imshow(image,cmap='gray')
# plt.show()
# print('Label:',label)


## Loading the Images as the torch Tensors:

# transforms = transforms.Compose(
#     [
#         transforms.Resize(28),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             [0.5 for _ in range(1)], [0.5 for _ in range(1)]
#         ),

#     ]
# )
dataset = torch_dataset.MNIST(root='data/',transform=transforms.ToTensor(),train=True)

img_tensor ,label = dataset[0]
# print(img_tensor.shape,label)

# The shape = [1, 28, 28] : 1 colour Channel and 28x28 pixels..
# print(torch.max(img_tensor),torch.min(img_tensor))# 1 represent black and 0 represents white.

# chopped_tensor = img_tensor[0,20:25,0:]
# print(chopped_tensor)
# plt.imshow(chopped_tensor,cmap='gray')
# plt.show()

## Splitting the Dataset into training Set Validation set and Test set.
# The MNIST data set has 60,000 training samples and 10,000 test.
# So we have to split the 60,000 samples into train and validation sets.

from torch.utils.data import random_split
length =(len(dataset))
# print(length*0.7)
# print(length*0.3)
train_ds,val_ds = random_split(dataset,[int(length*0.7),int(length*0.3)])
# print(len(train_ds),len(val_ds))

## Making the Batches:
BATCH_SZ = 140
train_loader = DataLoader(train_ds,batch_size = BATCH_SZ,shuffle=True)
val_loader = DataLoader(val_ds,batch_size=BATCH_SZ)# Not Necessary for Randomisation

# for img,label in train_loader:
#     print(img.shape,label)
#     # plt.imshow(img[0],cmap='gray')
#     break;

### MODEL!!
input_size = 1*28*28 # Flatten cause Linear layer takes vector
output_size = 10 # 0-9 classification
# model = nn.Linear(input_size,output_size)

# print(model.weight.shape,model.bias.shape)

## Instead of reshaping the Images over and over we can create a extended version of the nn.Module() class
## And Make our model as an object of that!!!!

def accuracy(probabilities,labels):
    _, preds = torch.max(probabilities,dim=1)
    return torch.tensor(torch.sum(preds==labels).item() / len(preds))

class ImgModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size,output_size),
            nn.Softmax(),# Probilities !!!! sum of all vals = 1.
        )

    def forward(self,xb):
        xb = xb.reshape(-1,784);
        out = self.layer(xb)
        return out
    
    def training_step(self, batch):
        images, labels = batch
        out = self.forward(images) # Generate Predictions
        lossfun = nn.CrossEntropyLoss() # Calculate Loss
        loss = lossfun(out,labels)
        return loss

    def validation_step(self,batch):
        images, labels = batch
        out = self.forward(images)
        lossfun = nn.CrossEntropyLoss()
        loss = lossfun(out, labels)
        acc = accuracy(out,labels)
        return {'val_loss': loss,'val_acc':acc}
    
    def validation_epoch_end(self,outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean() # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean() # combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self,epoch,result):
        print ("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch,result['val_loss'],result['val_acc']))

model = ImgModel();

# print(model.linear.weight.shape,model.linear.bias.shape)
# print(list(model.parameters()))


# for images,labels in train_loader:
#     print(images.shape)
#     probabilities = model(images)
#     print('Outputshape:',probabilities.shape)
#     # print('Samples:\n',pred[:1])
#     # sum = torch.sum(pred[:1])
#     # print(sum) # SUMS to 1
#     max_probs,preds = torch.max(probabilities,dim = 1)
#     # print(max_probs,preds)
#     break
  

## Training !!


loss_fn = nn.CrossEntropyLoss()

# for images,labels in train_loader:
#     probabilities = model(images)
#     acc = accuracy(probabilities,labels)
#     loss = loss_fn(probabilities,labels)
#     print(loss)
#     print(acc)
#     break

## TRAIN :

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

# EPOCH = 200
# LR = 1e-5
def train(EPOCH,LR,model,train_loader,val_loader,opt_func = optim.SGD):
    optimizer = opt_func(model.parameters(),lr = LR)
    history = [] # For recording epoch-wise results

    for epoch in range(EPOCH):

        # Training Phase:
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()  
        
        # Validation phase:
        result = evaluate(model,val_loader)
        model.epoch_end(epoch,result)
        history.append(result)

    return history

result0 = evaluate(model,val_loader)
print(result0)


history1 = train(5,0.001,model,train_loader,val_loader)
history2 = train(10,0.005,model,train_loader,val_loader)
history3 = train(15,0.01,model,train_loader,val_loader)









