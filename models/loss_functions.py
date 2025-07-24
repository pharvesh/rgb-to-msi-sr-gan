"""
Loss Functions for RGB to Multispectral Image Super-Resolution

This module implements various loss functions specifically designed for
spectral reconstruction tasks, including adversarial, reconstruction,
and spectral fidelity losses.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Optional, Tuple


class SpectralAngleMapperLoss(nn.Module):
    """
    Spectral Angle Mapper (SAM) Loss for spectral fidelity.
    
    SAM measures the angle between spectral vectors, providing a metric
    that is invariant to illumination changes while preserving spectral shape.
    
    """
    
    def __init__(self, 
                 epsilon: float = 1e-8, 
                 clamp_epsilon: float = 1e-6,
                 reduction: str = 'mean'):
        super().__init__()
        
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
        
        self.epsilon = epsilon
        self.clamp_epsilon = clamp_epsilon
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
      
        # Check for NaN or Inf values
        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
            print("Warning: NaN or Inf detected in SAMLoss input (y_pred). "
                  "Returning zero loss.", file=sys.stderr)
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        # Reshape to (N, C) where N = B*H*W
        y_pred_flat = y_pred.permute(0, 2, 3, 1).reshape(-1, y_pred.size(1))
        y_true_flat = y_true.permute(0, 2, 3, 1).reshape(-1, y_true.size(1))

        # Calculate L2 norms
        pred_norm = torch.linalg.norm(y_pred_flat, ord=2, dim=1)
        true_norm = torch.linalg.norm(y_true_flat, ord=2, dim=1)
        
        # Calculate dot product
        dot_product = (y_pred_flat * y_true_flat).sum(dim=1)
        
        # Calculate cosine similarity
        norm_product = pred_norm * true_norm
        cosine_similarity = torch.clamp(
            dot_product / (norm_product + self.epsilon),
            -1.0 + self.clamp_epsilon, 
            1.0 - self.clamp_epsilon
        )
        
        # Calculate angles
        angles = torch.acos(cosine_similarity)
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(angles)
        elif self.reduction == 'sum':
            return torch.sum(angles)
        else:
            return angles.reshape(y_pred.shape[0], y_pred.shape[2], y_pred.shape[3])


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained VGG features.
    
   
    """
    
    def __init__(self, 
                 feature_layers: list = [3, 8, 15, 22],
                 weights: Optional[list] = None):
        super().__init__()
        
        # Load pre-trained VGG19
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
        self.features = vgg.features
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.feature_layers = feature_layers
        self.weights = weights or [1.0] * len(feature_layers)
        
        if len(self.weights) != len(self.feature_layers):
            raise ValueError("Number of weights must match number of feature layers")

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
     
        # For multispectral images, use first 3 channels as RGB approximation
        if y_pred.size(1) > 3:
            y_pred_rgb = y_pred[:, :3, :, :]
            y_true_rgb = y_true[:, :3, :, :]
        else:
            y_pred_rgb = y_pred
            y_true_rgb = y_true
        
        # Normalize to ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).to(y_pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(y_pred.device).view(1, 3, 1, 1)
        
        y_pred_norm = (y_pred_rgb - mean) / std
        y_true_norm = (y_true_rgb - mean) / std
        
        # Extract features
        loss = 0.0
        x_pred = y_pred_norm
        x_true = y_true_norm
        
        for i, layer in enumerate(self.features):
            x_pred = layer(x_pred)
            x_true = layer(x_true)
            
            if i in self.feature_layers:
                layer_idx = self.feature_layers.index(i)
                weight = self.weights[layer_idx]
                loss += weight * F.mse_loss(x_pred, x_true)
        
        return loss


class GANLoss(nn.Module):
    """
    GAN loss with label smoothing and different loss types.
    
    """
    
    def __init__(self, 
                 gan_mode: str = 'vanilla',
                 target_real_label: float = 1.0,
                 target_fake_label: float = 0.0,
                 label_smoothing: bool = True):
        super().__init__()
        
        self.gan_mode = gan_mode
        self.real_label_val = target_real_label
        self.fake_label_val = target_fake_label
        self.label_smoothing = label_smoothing
        
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'wgangp':
            self.loss = None  # Wasserstein loss doesn't use a loss function
        else:
            raise ValueError(f"Unsupported GAN mode: {gan_mode}")

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
       
        if target_is_real:
            if self.label_smoothing:
                # Uniform random values between 0.8 and 1.0 for real labels
                target_tensor = torch.empty_like(prediction).uniform_(0.8, 1.0)
            else:
                target_tensor = torch.full_like(prediction, self.real_label_val)
        else:
            if self.label_smoothing:
                # Uniform random values between 0.0 and 0.2 for fake labels
                target_tensor = torch.empty_like(prediction).uniform_(0.0, 0.2)
            else:
                target_tensor = torch.full_like(prediction, self.fake_label_val)
        return target_tensor

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
       
        if self.gan_mode == 'wgangp':
            if target_is_real:
                return -prediction.mean()
            else:
                return prediction.mean()
        else:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            return self.loss(prediction, target_tensor)


class CombinedLoss(nn.Module):
    """
    Combined loss function for the generator incorporating multiple loss terms.
    
   
    """
    
    def __init__(self, 
                 lambda_l1: float = 100.0,
                 lambda_sam: float = 0.3,
                 lambda_adv: float = 0.1,
                 lambda_perceptual: float = 10.0,
                 use_perceptual: bool = False):
        super().__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_sam = lambda_sam
        self.lambda_adv = lambda_adv
        self.lambda_perceptual = lambda_perceptual
        self.use_perceptual = use_perceptual
        
        # Initialize loss functions
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.sam_loss = SpectralAngleMapperLoss()
        self.gan_loss = GANLoss()
        
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()

    def forward(self, 
                generated: torch.Tensor,
                target: torch.Tensor,
                discriminator_output: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        
        losses = {}
        
        # Reconstruction losses
        l1_loss = self.l1_loss(generated, target)
        sam_loss = self.sam_loss(generated, target)
        
        losses['l1'] = l1_loss
        losses['sam'] = sam_loss
        
        total_loss = self.lambda_l1 * l1_loss + self.lambda_sam * sam_loss
        
        # Adversarial loss
        if discriminator_output is not None:
            adv_loss = self.gan_loss(discriminator_output, target_is_real=True)
            losses['adversarial'] = adv_loss
            total_loss += self.lambda_adv * adv_loss
        
        # Perceptual loss
        if self.use_perceptual:
            perceptual_loss = self.perceptual_loss(generated, target)
            losses['perceptual'] = perceptual_loss
            total_loss += self.lambda_perceptual * perceptual_loss
        
        losses['total'] = total_loss
        
        return total_loss, losses


def gradient_penalty(discriminator: nn.Module,
                    real_rgb: torch.Tensor,
                    real_spectral: torch.Tensor,
                    fake_spectral: torch.Tensor,
                    device: torch.device,
                    lambda_gp: float = 10.0) -> torch.Tensor:
    """
    Calculate gradient penalty for WGAN-GP.
   
    """
    batch_size = real_spectral.size(0)
    
    # Random interpolation factor
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    
    # Interpolate between real and fake spectral images
    interpolated = alpha * real_spectral + (1 - alpha) * fake_spectral
    interpolated.requires_grad_(True)
    
    # Get discriminator output for interpolated images
    d_interpolated = discriminator(real_rgb, interpolated)
    
    # Calculate gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Calculate gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty