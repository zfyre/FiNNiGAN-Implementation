import os
import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from model2 import Discriminator, UNET
from FinniDataset import FinniGANDataset


# Hyperparams
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Lr = 1e-5
Batch_size = 100
Image_size = 160
Channel_img = 3
Num_Epochs = 5
Features_Disc = 160
Features_Gen = 160

# Appling the Transforms
CustomTransform = transforms.Compose(
    [
        transforms.Resize((Image_size,Image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(Channel_img)], [0.5 for _ in range(Channel_img)]
        )
    ]
)
# Load Data
dataset = FinniGANDataset(root_dir = os.path.join('data', 'output'), transform=CustomTransform)

train_set, test_set = torch.utils.data.random_split(dataset, [486,200])
train_loader = DataLoader(dataset=train_set, batch_size=Batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=Batch_size, shuffle=True)

# for batch_idx, (imgIn,imgOut) in enumerate(train_loader):
#     print(imgIn.shape, imgOut.shape)
# t1,t2,I1,I2,O = train_set[0]

# async def ShowImg(idx):
fig = plt.figure(figsize=(10,7))
rows = 1
columns = 3
t1,t2,I1,I2,O = train_set[0]
print(t1.shape,t2.shape)
# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)

# showing image
plt.imshow(I1.permute(1,2,0))
# plt.axis('off')
plt.title("First")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)

# showing image
plt.imshow(O.permute(1,2,0))
# plt.axis('off')
plt.title("Second")

# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)

# showing image
plt.imshow(I2.permute(1,2,0))
# plt.axis('off')
plt.title("Third")


# ShowImg(0)