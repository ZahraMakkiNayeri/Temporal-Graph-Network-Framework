import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Multiclass focal loss with per-class weighting.

    FIX (Reviewer 6, Comment 4): the previous implementation applied the
    scalar alpha only to samples with target == 1 — a binary-focal-loss
    idiom that silently mis-weighted the multiclass problem (only one of
    the four categories was down-weighted). `alpha` may now be:
      * None              -> no class weighting;
      * float             -> uniform scaling (legacy behaviour, all classes);
      * sequence / tensor -> per-class weights of length num_classes
                             (recommended: computed on the TRAINING split
                             when BALANCE_CLASSES is False).
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (list, tuple)):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logpt = -F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(logpt)
        loss = ((1 - pt) ** self.gamma) * (-logpt)

        if torch.is_tensor(self.alpha):
            loss = loss * self.alpha.to(inputs.device)[targets]
        elif self.alpha is not None:
            loss = loss * self.alpha

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
