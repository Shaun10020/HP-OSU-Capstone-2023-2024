# importing necesary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models.segmentation as segmentation

class CustomDeepLabV3(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomDeepLabV3, self).__init__()
        # Initialize DeepLabV3 with a ResNet-50 backbone
        self.model = segmentation.deeplabv3_resnet50(pretrained=False, num_classes=num_classes)

        # Modify the first convolutional layer to accept 5-channel input
        self.model.backbone.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        # DeepLabV3 forward pass
        output = self.model(x)['out']

        return output

