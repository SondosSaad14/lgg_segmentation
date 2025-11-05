import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth: float=1.0):
        super().__init__(); self.smooth=smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2.0*(probs*targets).sum(dim=(2,3)) + self.smooth
        den = (probs + targets).sum(dim=(2,3)) + self.smooth
        dice = (num/den).mean()
        return 1.0 - dice

def bce_dice_loss(logits, targets):
    return F.binary_cross_entropy_with_logits(logits, targets) + DiceLoss()(logits, targets)

