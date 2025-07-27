"""
Loss Functions for RGB to Multispectral Image Super-Resolution

This module implements various loss functions specifically designed for
spectral reconstruction tasks, including adversarial, reconstruction,
and spectral fidelity losses.

"""

import torch
import torch.nn as nn
import sys

class SAMLoss(nn.Module):
    def __init__(self, epsilon=1e-8, clamp_eps=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.clamp_eps = clamp_eps

    def forward(self, y_pred, y_true):
        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
            print("NaN/Inf in SAMLoss input", file=sys.stderr)
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        y_p = y_pred.permute(0,2,3,1).reshape(-1, y_pred.size(1))
        y_t = y_true.permute(0,2,3,1).reshape(-1, y_true.size(1))

        pn = torch.linalg.norm(y_p, dim=1)
        tn = torch.linalg.norm(y_t, dim=1)
        dp = (y_p * y_t).sum(dim=1)
        cosine = torch.clamp(dp / (pn * tn + self.epsilon),
                             -1.0 + self.clamp_eps, 1.0 - self.clamp_eps)
        return torch.mean(torch.acos(cosine))



class GANLoss(nn.Module):
    """ Wrapper around BCEWithLogitsLoss for label-target adversarial training."""
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, prediction, target_label):
        """
        target_label: Tensor of same shape as prediction, with values 0 (fake) or 1 (real)
        """
        return self.loss(prediction, target_label)

def reconstruction_l1(pred, target):
    return nn.L1Loss(reduction='mean')(pred, target)























