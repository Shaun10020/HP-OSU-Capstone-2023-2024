import torch

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
