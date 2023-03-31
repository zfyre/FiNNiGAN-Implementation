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
from model import Discriminator, UNET, initialize_weights
from FinniDataset import FinniGANDataset
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

testModel = UNET(3,3)

testModel.load_state_dict(torch.load("C:/Users/98107/Desktop/BYOP/pytorch/GANs/logs/gen1.pth"),strict=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams:
Lr = 2e-3
Batch_size = 100
Image_size = 256 # To be taken Care of !!
Channel_img = 3
Num_Epochs = 5
Stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) 

# Appling the Transforms:
CustomTransform = transforms.Compose(
    [
        transforms.Resize((Image_size,Image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(Channel_img)], [0.5 for _ in range(Channel_img)]
        )
    ]
)
# Load Data:
dataset = FinniGANDataset(root_dir = os.path.join('data', 'output'), transform=CustomTransform)
datalen = len(dataset)

train_set, test_set = torch.utils.data.random_split(dataset, [int(0.8*datalen),datalen-int(0.8*datalen)])

def denorm(img_tensor):
    return img_tensor*Stats[1][0] + Stats[0][0]

def showImg(idx):
    t1,t2,Img1,Img2 = train_set[idx]
    T1 = torch.unsqueeze(t1,dim=0)
    print(t1.shape,t2.shape)
    genout = testModel(T1)
    genout = torch.squeeze(genout,dim=0)
    print(genout.shape,torch.min(genout),torch.max(genout))
    genout = genout.permute(1,2,0)
    genout = genout.detach().numpy()
    f , axarr = plt.subplots(1,4)
    axarr[0].imshow(denorm(Img1).permute(1,2,0))
    axarr[1].imshow(denorm(t2).permute(1,2,0))
    axarr[2].imshow(denorm(genout))
    axarr[3].imshow(denorm(Img2).permute(1,2,0))
    f.show()
    plt.show(block=True)

for i in range(len(dataset)):
    showImg(i)
    # break