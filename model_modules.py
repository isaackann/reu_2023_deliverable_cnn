import torch
import torch.nn as nn


""" Isaac Kan 7.27.23
    ----------------------------------------------------------------------------------------------
    This file contains the implementation of a U-Net style convolutional neural network for 
    MRI super-resolution. The network takes in 2 low-res Na MRI scans and a high-res 
    Proton MRI scan to produce a high-res AGR of the Na MRI.
    
    This model accepts and reproduces 3D images.
    
    Follow this link to see model architecture: https://imgur.com/a/q5dntpD 
"""


#-------------------------------------------------------------------------------------------------
# Modules used to build the U-Net network


class DoubleConv3d(nn.Module):
    """ 
    Double convolution pipeline that increases/decreases the number of features 
     - [Conv2d -> BatchNorm -> ReLU] x2 
    """
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        
        # Only up-convolution requires the mid-channel parameter
        if not mid_channels: mid_channels = out_channels
        
        self.double_conv_3d = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)            
        )
        
    def forward(self, x):
        return self.double_conv_3d(x)
    

class SendDown(nn.Module):
    """ 
    Module used in the encoding half of the U-Net. 
     - Image size is decreased with 2x2 max-pooling 
     - Double convolution is applied to double the number of features 
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.maxpool_and_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DoubleConv3d(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_and_conv(x)
    

class SendUp(nn.Module):
    """ 
    Module used in the decoding half of the U-Net. 
     - Image size is doubled with upsampling
     - Double convolution is applied to halve the number of features 
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.upscale = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv3d = DoubleConv3d(in_channels, out_channels, mid_channels=in_channels // 2)
    
    def forward(self, lower, left):
        lower = self.upscale(lower)  
        
        # Ensure that the lower upsample is the same size as the opposite input for concatenation
        deltaX, deltaY, deltaZ = (left.size()[i] - lower.size()[i] for i in range(2, 5))
        lower = nn.functional.pad(lower, [deltaX // 2, deltaX - deltaX // 2, 
                                          deltaY // 2, deltaY - deltaY // 2, 
                                          deltaZ // 2, deltaZ - deltaZ // 2])
        
        # Concatenate tensors along channel dimension, then feed through single convolutional layer
        return self.conv3d(torch.cat((lower, left), dim=1))


#-------------------------------------------------------------------------------------------------
# Implementation of U-Net CNN

class SuperResUNetCNN(nn.Module):
    """ 
    Class implementation of CNN
    See architecture visualization here: https://imgur.com/a/q5dntpD
    """

    def __init__(self):
        super().__init__()
        
        # Sequence of encoding modules to reduce nxn input 
        self.input_layer = DoubleConv3d(3, 64)  # increase from initial # of channels to 64 channels
        self.down1, self.down2, self.down3 = SendDown(64, 128), SendDown(128, 256), SendDown(256, 256)  
        
        # Sequence of decoding modules to reconstruct the image
        self.up1, self.up2, self.up3 = SendUp(512, 128), SendUp(256, 64), SendUp(128, 64)
        self.output_layer = nn.Conv3d(64, 1, kernel_size=1)  # convolutional layer to produce final image
        
    
    def forward(self, concat_tensor):
        # 3-channel concatenated tensor -> 64-channel convolution
        input_conv = self.input_layer(concat_tensor)
        
        # Encoding: reduce image size 8x and increase channel dimension 4x
        down_layer_1 = self.down1(input_conv)
        down_layer_2 = self.down2(down_layer_1)
        down_layer_3 = self.down3(down_layer_2)
        
        # Decoding: increase image size 8x and decrease channel dimension 4x
        up_layer_1 = self.up1(down_layer_3, down_layer_2)
        up_layer_2 = self.up2(up_layer_1, down_layer_1)
        up_layer_3 = self.up3(up_layer_2, input_conv)
        
        return self.output_layer(up_layer_3)  # Final layer to produce output image
