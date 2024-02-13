import torch
from config.config import threshold

def convertBinary(preds):
    # preds = torch.sigmoid(preds)
    assert(torch.where(preds>1.0,1.0,0.0).sum()==0)
    assert(torch.where(preds<0.0,1.0,0.0).sum()==0)
    assert(torch.where(torch.isnan(preds),1.0,0.0).sum()==0)
    return torch.where(preds > threshold,1.0,0.0)