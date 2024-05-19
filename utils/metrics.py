import torch
import torch.nn as nn

def binary_iou(pred, target):
    """
    Calculate the Intersection over Union (IoU) for binary segmentation.
    
    Args:
    pred (torch.Tensor): Predicted bitmap (after thresholding if necessary).
    target (torch.Tensor): Ground truth bitmap.

    Returns:
    float: IoU score.
    """
    intersection = torch.logical_and(pred, target).sum().float()
    union = torch.logical_or(pred, target).sum().float()

    if union == 0:
        # To avoid division by zero
        return 1.0
    else:
        iou = intersection / union
        return iou.item()


# Creating as a placeholder for calculating Dice Loss
# Need to check for accuracy

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce_losss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):

        BCE = self.bce_losss(inputs, targets)

        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(
            inputs.sum() + targets.sum() + smooth)
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

def accuracy(pred,target):
    """
    Calculate the Accuracy for binary segmentation.
    
    Args:
    pred (torch.Tensor): Predicted bitmap (after thresholding if necessary).
    target (torch.Tensor): Ground truth bitmap.

    Returns:
    float: Accuracy
    """
    pred = pred.bool()
    target = target.bool()
    correct = (pred == target).float().sum()
    incorrect = (pred != target).float().sum()
    
    return (correct / (correct + incorrect)).item()

def precision(pred,target):
    """
    Calculate the Precision for binary segmentation.
    
    Args:
    pred (torch.Tensor): Predicted bitmap (after thresholding if necessary).
    target (torch.Tensor): Ground truth bitmap.

    Returns:
    float: Precision
    """
    intersection = torch.logical_and(pred, target).sum().float()
    false_positive = torch.logical_and(pred.bool(), ~target.bool()).sum().float()
    if intersection.item():
        return intersection/(intersection + false_positive)
    else:
        return torch.zeros(1)

def recall(pred,target):
    """
    Calculate the Recall for binary segmentation.
    
    Args:
    pred (torch.Tensor): Predicted bitmap (after thresholding if necessary).
    target (torch.Tensor): Ground truth bitmap.

    Returns:
    float: Recall
    """
    intersection = torch.logical_and(pred, target).sum().float()
    false_negative = torch.logical_and(~pred.bool(),target.bool()).sum().float()
    if intersection.item():
        return intersection/(intersection + false_negative)
    else:
        return torch.zeros(1)

def f1_score(pred,target):
    """
    Calculate the F1 score for binary segmentation.
    
    Args:
    pred (torch.Tensor): Predicted bitmap (after thresholding if necessary).
    target (torch.Tensor): Ground truth bitmap.

    Returns:
    float: F1 score
    """
    _precision = precision(pred,target).item()
    _recall = recall(pred,target).item()
    if (_precision+_recall) == 0.0:
        return _precision,_recall,0.0
    return _precision,_recall,(2*_precision*_recall)/(_precision+_recall)
