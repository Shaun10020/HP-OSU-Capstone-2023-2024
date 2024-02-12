import torch
from config.config import threshold

def convertBinary(preds):
    preds = torch.sigmoid(preds)
    return torch.where(preds > threshold,1.0,0.0)