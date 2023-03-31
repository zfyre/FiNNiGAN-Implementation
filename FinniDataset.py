import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage import io  
import cv2

def stack (img1,img2):
    return torch.div(torch.add(img1,img2),2)

class FinniGANDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        directory = os.fsencode(self.root_dir)
        return len(os.listdir(directory))
    
    def __getitem__(self,index):
        img1_path = os.path.join(self.root_dir,'data'+str(index),'frame1.png') # The directory name of each sample would be data0, data1, etc..
        img2_path = os.path.join(self.root_dir,'data'+str(index),'frame3.png')
        orig_img_path = os.path.join(self.root_dir,'data'+str(index),'frame2.png')
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        orig_img = Image.open(orig_img_path)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            orig_img = self.transform(orig_img)
        
        # print(img1.shape, img2.shape)
        img_stk = stack(img1, img2)

        return (img_stk, orig_img, img1, img2)
        

# The transform Can be transforms.ToTensor()