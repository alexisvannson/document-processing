import torch.nn as nn
import torch.nn.functional as F

class DBLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=10.0, ohem_ratio=3):
        super().__init__()
        self.alpha = alpha # Weight for Binary map loss (Dice)
        self.beta = beta   # Weight for Threshold map loss (L1)
        self.ohem_ratio = ohem_ratio

    def ohem_bce(self, pred, target, mask):
        """ BCE Loss that only calculates gradient for the hardest negative examples """
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        
        # Apply ignore mask (e.g., regions where text is illegible)
        bce_loss = bce_loss * mask
        
        # Separate positives and negatives
        positives = bce_loss[target > 0.5]
        negatives = bce_loss[target <= 0.5]
        
        # Keep all positives, but only keep the hardest negatives (highest loss)
        num_pos = positives.numel()
        num_neg = min(int(num_pos * self.ohem_ratio), negatives.numel())
        
        if num_pos == 0 or num_neg == 0:
            return bce_loss.mean()
            
        hard_negatives, _ = negatives.topk(num_neg)
        
        # Average loss over positives and hard negatives
        return (positives.sum() + hard_negatives.sum()) / (num_pos + num_neg)

    def dice_loss(self, pred, target, mask):
        """ Dice loss ensures the predicted text shape matches the target """
        pred = pred * mask
        target = target * mask
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)

    def masked_smooth_l1(self, pred, target, text_border_mask):
        """ Smooth L1 is only calculated inside the dilated text borders """
        loss = F.smooth_l1_loss(pred, target, reduction='none')
        return (loss * text_border_mask).sum() / (text_border_mask.sum() + 1e-5)

    def forward(self, preds, gt_prob, gt_thresh, border_mask, ignore_mask):
        pred_prob, pred_thresh, pred_binary = preds
        
        # 1. Probability Map Loss (BCE + OHEM)
        l_s = self.ohem_bce(pred_prob, gt_prob, ignore_mask)
        
        # 2. Approximate Binary Map Loss (Dice)
        l_b = self.dice_loss(pred_binary, gt_prob, ignore_mask)
        
        # 3. Threshold Map Loss (Masked L1)
        l_t = self.masked_smooth_l1(pred_thresh, gt_thresh, border_mask)
        
        # Total Loss Equation from the paper
        total_loss = l_s + (self.alpha * l_b) + (self.beta * l_t)
        
        return total_loss, {"loss_prob": l_s, "loss_binary": l_b, "loss_thresh": l_t}
    
