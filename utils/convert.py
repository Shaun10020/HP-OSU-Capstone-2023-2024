import torch
from config.config import threshold

def convertBinary(preds):
    '''
    This function converts grayscale image to binary mask
    
    param preds: given a predicted tensors
    
    Return:
    array of tensors where each element is 1.0 or 0.0
    '''
    
    ## assert to make sure the predicted values do not goes over the range
    assert(torch.where(preds>1.0,1.0,0.0).sum()==0)
    assert(torch.where(preds<0.0,1.0,0.0).sum()==0)
    assert(torch.where(torch.isnan(preds),1.0,0.0).sum()==0)
    
    
    return torch.where(preds > threshold,1.0,0.0)