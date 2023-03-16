"""
The Generator and the Dicriminator Classes for the model:
"""

import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            # Input : N x (6,sz,sz)
            self._block_C(6,32),
            self._block_C(32,64),
            self._block_C(64,64),
            self._block_C(64,128),
            self._block_D(128,128),
            self._block_D(128,64),
            self._block_D(64,64),
            self._block_D(64,32),
        )
    ## K : Number of Filters
    """
    if inchannels = I,Number of filters = K and Kernel size and stride and padding are as in the functions:
    (I,sz,sz)-->_block_C-->()
    """
    def _block_C(self, in_channels,K,kernel_size = 3, stride = 1, padding = 0):
        return nn.Sequential(
            nn.Conv2d(in_channels,K,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(K),
            nn.LeakyReLU(0.2),
            nn.Conv2d(K,K,kernel_size,stride,padding),
            nn.BatchNorm2d(K),
            nn.LeakyReLU(0.2),
    )
    def _block_D(self, in_channels,K,kernel_size = 4, stride = 1, padding = 0):
        return nn.Sequential(
            # nn.Bilinear(),
            nn.Conv2d(in_channels,K,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(K),
            nn.LeakyReLU(0.2),
            nn.Conv2d(K,K,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(K),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.5),  
    )
    def forward(self,x):
        return self.gen(x)

testmodel = Generator()
summary(testmodel,input_size=(6,64,64))

"""
Under Progress!!
"""