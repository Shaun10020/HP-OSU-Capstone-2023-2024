from .modules.UNet_parts import *
import logging

'''
This is the Unet or Unet realated architectures
'''
class UNet(nn.Module):
    """UNet model

    Args:
        number_features (int): This the number of input channels 
        number_claasses (int): This the number of output channels
    """
    
    def __init__(self,number_features,number_classes):
        logging.info("Initializing UNet Model...")
        super().__init__()
        self.number_features = number_features
        self.number_classes = number_classes
        
        self.inputConv = DoubleConv(number_features,64)
        self.downSampling1 = DownSampling(64, 128)
        self.downSampling2 = DownSampling(128, 256)
        self.downSampling3 = DownSampling(256, 512)
        self.downSampling4 = DownSampling(512, 1024)
        
        self.upSampling1 = UpSampling(1024, 512)
        self.upSampling2 = UpSampling(512, 256)
        self.upSampling3 = UpSampling(256, 128)
        self.upSampling4 = UpSampling(128, 64)
        
        self.outputConv = OutputConv(64,number_classes)
        logging.info("Done initialize UNet Model")
        
    def forward(self,x):
        x1 = self.inputConv(x)
        x2 = self.downSampling1(x1)
        x3 = self.downSampling2(x2)
        x4 = self.downSampling3(x3)
        x5 = self.downSampling4(x4)
        x = self.upSampling1(x5, x4)
        x = self.upSampling2(x, x3)
        x = self.upSampling3(x, x2)
        x = self.upSampling4(x, x1)
        logits = self.outputConv(x)
        return logits
    
    
class CustomUNet1(nn.Module):
    """Custom UNet model

    Args:
        number_features (int): This the number of input channels 
        number_claasses (int): This the number of output channels
    """
    
    def __init__(self,number_features,number_classes):
        logging.info("Initializing UNet Model...")
        super().__init__()
        self.number_features = number_features
        self.number_classes = number_classes
        
        self.inputConv = DoubleConv(number_features,64)
        self.downSampling1 = DownSampling(64, 128)
        self.downSampling2 = DownSampling(128, 256)
        self.downSampling3 = DownSampling(256, 512)
        
        self.upSampling1 = UpSampling(512, 256)
        self.upSampling2 = UpSampling(256, 128)
        self.upSampling3 = UpSampling(128, 64)
        
        self.outputConv = OutputConv(64,number_classes)
        logging.info("Done initialize UNet Model")
        
    def forward(self,x):
        x1 = self.inputConv(x)
        x2 = self.downSampling1(x1)
        x3 = self.downSampling2(x2)
        x4 = self.downSampling3(x3)
        x = self.upSampling1(x4, x3)
        x = self.upSampling2(x, x2)
        x = self.upSampling3(x, x1)
        logits = self.outputConv(x)
        return logits