import os
import cv2
import torch 
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from model2 import Discriminator, UNET, initialize_weights

Image_size = 256 # To be taken Care of !!
Channel_img = 3
Stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denorm(img_tensor):
    return img_tensor*Stats[1][0] + Stats[0][0]

def stack (img1,img2):
    return torch.div(torch.add(img1,img2),2)

CustomTransform = transforms.Compose(
    [
        transforms.Resize((Image_size,Image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(Channel_img)], [0.5 for _ in range(Channel_img)]
        )
    ]
)

gen = UNET(3,3)
gen.load_state_dict(torch.load(os.path.join('logs','WGenNoise.pth'),device),strict=False)

VID_directory_path = os.path.join('data','VidTest')
OUTPATH = os.path.join('data','outTest')

directory = os.fsencode(VID_directory_path)

def Frame_Interpolate(filename,cnt):   
    cap = cv2.VideoCapture(os.path.join(VID_directory_path,filename))
    # Setup Video Writer:
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videowriter = cv2.VideoWriter(os.path.join('data','outTest','output'+ str(outcount) +'.avi'),fourcc,2*fps,(256,256),isColor = True)
    for frame_idx in range((int(cap.get(cv2.CAP_PROP_FRAME_COUNT))//2)*2 - 1):

        # Read Frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success,f1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx+1)
        success,f2 = cap.read()
        # Convert to numpy array
        f1 = Image.fromarray(f1)
        f2 = Image.fromarray(f2)
        # print(f1)

        # Convert to Tensor
        f1 = CustomTransform(f1)
        f2 = CustomTransform(f2)

        # Generate the Images:
        middle = stack(f1,f2)
        middle = torch.unsqueeze(middle,dim=0)
        genout = gen(middle)
        genout = torch.squeeze(genout,dim=0)

        # Normalize
        f1 = denorm(f1)
        genout = denorm(genout)
        
        # Convert to Numpy
        f1=torch.Tensor.cpu(f1).detach().permute(1,2,0).numpy()
        f2=torch.Tensor.cpu(f2).detach().permute(1,2,0).numpy()
        genout=torch.Tensor.cpu(genout).detach().permute(1,2,0).numpy()

        # Correct the channels:
        # genout = cv2.cvtColor(genout,cv2.COLOR_BGR2RGB)

        # Convert to uint8 form:
        f1 = cv2.normalize(f1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        genout = cv2.normalize(genout, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        cv2.imshow('Video Player',genout)
        # Write the frame:
        videowriter.write(f1)
        videowriter.write(genout)
        # videowriter.write(f2).
        # Render:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close down Everything and release Video Writer:
    cap.release()
    cv2.destroyAllWindows()
    videowriter.release()


outcount = 0
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    file_path = os.path.join(VID_directory_path,filename)
    Frame_Interpolate(filename,outcount)
    print(filename)
    outcount+=1
