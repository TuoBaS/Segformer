from torch import nn
from torch.nn import functional as F

from .Score import dice_score


class ADE20KDataSetLoss(nn.Module):
    def __init__(self, ce_weight=0.4, dice_weight=0.6, ignore_index=255):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

    def forward(self, predictions, targets):
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        ce_loss = self.ce(predictions, targets)

        pred_softmax = F.softmax(predictions, dim=1)
        valid_mask = targets != self.ignore_index
        safe_targets = targets.clamp(0, predictions.shape[1] - 1)
        targets_onehot = F.one_hot(safe_targets, predictions.shape[1]).permute(0, 3, 1, 2).float()

        valid_mask = valid_mask.unsqueeze(1).float()
        pred_softmax = pred_softmax * valid_mask
        targets_onehot = targets_onehot * valid_mask

        dice_loss = 1.0 - dice_score(
            pred_softmax,
            targets_onehot,
            reduce_batch=True,
            ignore_background=False,
        )

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
