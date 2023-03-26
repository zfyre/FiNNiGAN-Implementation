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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams:
Lr = 2e-4
Batch_size = 100
Image_size = 256 # To be taken Care of !!
Channel_img = 3
Num_Epochs = 5
# Features_Disc = 160
# Features_Gen = 160
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
train_loader = DataLoader(dataset=train_set, batch_size=Batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=Batch_size, shuffle=True)

# Initializing The Model:
gen = UNET(in_channels=3,out_channels=3).to(device)
disc = Discriminator(in_channels=3).to(device)
# initialize_weights(gen)
# initialize_weights(disc)

# Using different optimisers for gen and disc:
opt_gen = optim.Adam(gen.parameters(), lr=Lr, betas=(0.5,0.999)) # Betas Can be changed:
opt_disc = optim.Adam(disc.parameters(), lr=Lr, betas=(0.5,0.999)) # Betas Can be changed:

# Defining the Loss for Generator and Discriminator:

binary_cross_entropy = nn.BCELoss()
l1_loss = nn.L1Loss()
clipping_loss = nn.MSELoss()

# Telling the activation and other optimization layers to work:
gen.train()
disc.train()


def denorm(img_tensor):
    return img_tensor*Stats[1][0] + Stats[0][0]

# def showImg(idx):
#     t1,t2,Img1,Img2 = train_set[idx]
#     # print(torch.min(t1))
#     T1 = torch.unsqueeze(t1,dim=0)
#     print(t1.shape,t2.shape)
#     genout = gen(T1)
#     genout = torch.squeeze(genout,dim=0)
#     print(genout.shape,torch.min(genout),torch.max(genout))
#     genout = genout.permute(1,2,0)
#     genout = genout.detach().numpy()
#     f , axarr = plt.subplots(1,4)
#     axarr[0].imshow(denorm(Img1).permute(1,2,0))
#     axarr[1].imshow(denorm(t2).permute(1,2,0))
#     axarr[2].imshow(denorm(genout))
#     axarr[3].imshow(denorm(Img2).permute(1,2,0))
#     # axarr[1,0].imshow(t1.permute(1,2,0))
#     # axarr[1,1].imshow(t2.permute(1,2,0))
#     f.show()
#     plt.show(block=True)

# showImg(0)
# for i in range(len(dataset)):
#     showImg(i)
#     break

# """I'll have to implement a show image function which shows a buch of images of a batch in one place in collab."""


# Training the Discriminator:
def train_discriminator(f1_f3_images,real_images, opt_d):
    # Clear Discriminator gradients
    opt_d.zero_grad()

    # Pass real images through discriminator
    real_preds = disc(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = binary_cross_entropy(real_preds,real_targets)
    real_score = torch.mean(real_preds).item()

    # Generate fake images
    fake_images = gen(f1_f3_images)

    # Pass fake images through dicriminator
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = disc(fake_images)
    fake_loss = binary_cross_entropy(fake_preds,fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()

    return loss.item(), real_score, fake_score

# Training the Generator:
# def stack(img1,img2,img3):

def ms_ssim(pred,orignal):
    pred = (pred+1)/2
    orignal = (orignal+1)/2
    ms_ssim_loss = 1-ms_ssim( pred, orignal, data_range=1, size_average=True )# See if clipping Helps!!! that is if data_range = 1 helps
    return ms_ssim_loss
         
def train_generator(f1_f3_images,real_images,opt_g):
    # Clear generator gradients
    opt_g.zero_grad()

    # Generate fake images
    fake_images = gen(f1_f3_images)

    # Try to fool the discriminator
    preds = disc(fake_images)

    # Losses:
    # Discriminator Loss:
    targets = torch.ones(Batch_size, 1, device=device)
    DISC_LOSS = binary_cross_entropy(preds, targets)

    # l1 Loss:
    L1_LOSS = l1_loss(denorm(fake_images),denorm(real_images))

    # MS-SSIM Loss:
    PREDclippedIMG = transforms.Normalize(Stats)(preds)
    MS_SSIM_LOSS = ms_ssim(real_images,PREDclippedIMG)

    # Clipping Loss:
    CLIPPING_LOSS = clipping_loss(PREDclippedIMG,preds)

    # Update generator weights
    loss = DISC_LOSS.item() + L1_LOSS.item() + MS_SSIM_LOSS.item() + CLIPPING_LOSS.item()
    loss.backward()
    opt_g.step()

    return loss.item()


# Training :
def train(epochs,lr,start_idx=1):
    if torch.cuda.is_available:
        torch.cuda.empty_cache()

    # Losses & scores:
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    for epoch in range(epochs):
        for batch_idx,(stkIMG,middleIMG,_,_) in enumerate(train_loader):
            stkIMG, middleIMG = stkIMG.to(device), middleIMG.to(device)
            # Train Discriminator: 
            loss_d, real_score, fake_score = train_discriminator(stkIMG,middleIMG,opt_disc)
            # Train generator:
            loss_g = train_generator(stkIMG,middleIMG,opt_gen)

        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        # Log losses & scores (last batch)
        print ("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_scores: {:.4f}, fake_scores: {:.4f}",epoch+1,epochs,loss_g,loss_d,real_score,fake_score)

    return losses_d, losses_d, real_scores, fake_scores


train(Num_Epochs,Lr)