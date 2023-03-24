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
from model import Discriminator, UNET
from FinniDataset import FinniGANDataset


# Hyperparams
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Lr = 1e-5
Batch_size = 100
Image_size = 512 # To be taken Care of !!
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
datalen = len(dataset)
print(datalen)
train_set, test_set = torch.utils.data.random_split(dataset, [int(0.8*datalen),int(datalen-0.8*datalen)])
train_loader = DataLoader(dataset=train_set, batch_size=Batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=Batch_size, shuffle=True)

# for batch_idx, (imgIn,imgOut) in enumerate(train_loader):
#     print(imgIn.shape, imgOut.shape)
gen = UNET(in_channels=6,out_channels=3)

def showImg(idx):
    t1,t2,I1,I2,O = train_set[idx]
    T1 = torch.unsqueeze(t1,dim=0)
    print(t1.shape,t2.shape)
    genout = gen(T1)
    genout = torch.squeeze(genout,dim=0)
    print(genout.shape)
    genout = genout.permute(1,2,0)
    genout = genout.detach().numpy()
    f , axarr = plt.subplots(2,3)
    axarr[0,0].imshow(I1.permute(1,2,0))
    axarr[0,1].imshow(O.permute(1,2,0))
    axarr[0,2].imshow(I2.permute(1,2,0))
    # axarr[1,0].imshow(t1.permute(1,2,0))
    axarr[1,1].imshow(t2.permute(1,2,0))
    axarr[1,2].imshow(genout)
    f.show()
    plt.show(block=True)

# showImg(0)
for i in range(len(dataset)):
    showImg(i)