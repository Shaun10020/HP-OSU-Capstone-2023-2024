import torch.nn as nn
import torch
import torch.nn.functional as F

class DoubleConv(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 dimension=None,
                 kernel_size = 3,
                 padding = 1,
                 bias=False):
        
        super().__init__()
        if not dimension:
            dimension = out_channels
        self.doubleConv = nn.Sequential(
            nn.Conv2d(in_channels,dimension,kernel_size = kernel_size, padding = padding,bias=bias),
            nn.BatchNorm2d(dimension),
            nn.ReLU(inplace=True),
            nn.Conv2d(dimension,out_channels,kernel_size = kernel_size, padding = padding,bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self,x):
        return self.doubleConv(x)
    
    
class DownSampling(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels):
        
        super().__init__()
        self.downSampling = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels,out_channels)
        )
        
    def forward(self,x):
        return self.downSampling(x)
    
class UpSampling(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 2, 
                 stride = 2):
        
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size = kernel_size, stride = stride)
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self,x1,x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutputConv(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        
    def forward(self,x):
        return self.conv(x)