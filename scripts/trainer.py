"""
Training module for RGB to Multispectral Image Super-Resolution GAN

This module implements the training loop with proper GAN training dynamics,
learning rate scheduling, and comprehensive logging.

Author: Your Name
Date: 2024
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import logging
from pathlib import Path
import json
import time
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.losses import CombinedLoss, GANLoss
from scripts.early_stopping import EarlyStopping


class GANTrainer:
    
    
    def __init__(self,
                 generator: nn.Module,
                 discriminator: nn.Module,
                 train_loader,
                 val_loader,
                 config: dict,
                 device: torch.device,
                 logger: Optional[logging.Logger] = None):
        
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # Extract configuration
        self.training_config = config['training']
        self.model_config = config['model']
        self.checkpoint_config = config.get('checkpoint', {})
        
        # Initialize optimizers
        self._setup_optimizers()
        
        # Initialize loss functions
        self._setup_loss_functions()
        
        # Initialize learning rate schedulers
        self._setup_schedulers()
        
        # Initialize early stopping
        self._setup_early_stopping()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_g_loss': [],
            'train_d_loss': [],
            'train_l1_loss': [],
            'train_sam_loss': [],
            'val_g_loss': [],
            'val_l1_loss': [],
            'val_sam_loss': [],
            'learning_rates': []
        }
        
        # Enable anomaly detection for debugging
        if config.get('debug', {}).get('detect_anomaly', False):
            torch.autograd.set_detect_anomaly(True)
    
    def _setup_optimizers(self):
        """Initialize optimizers for generator and discriminator."""
        optimizer_config = self.training_config.get('optimizer', {})
        
        # Generator optimizer
        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.training_config['initial_lr_g'],
            betas=optimizer_config.get('betas', (0.5, 0.999)),
            weight_decay=optimizer_config.get('weight_decay', 0.0)
        )
        
        # Discriminator optimizer
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.training_config['initial_lr_d'],
            betas=optimizer_config.get('betas', (0.5, 0.999)),
            weight_decay=optimizer_config.get('weight_decay', 0.0)
        )
    
    def _setup_loss_functions(self):
        """Initialize loss functions."""
        # Combined generator loss
        self.generator_loss = CombinedLoss(
            lambda_l1=self.training_config['lambda_l1'],
            lambda_sam=self.training_config['lambda_sam'],
            lambda_adv=self.training_config['lambda_adv'],
            use_perceptual=self.training_config.get('use_perceptual', False)
        ).to(self.device)
        
        # Discriminator loss
        gan_mode = self.training_config.get('gan_mode', 'vanilla')
        self.discriminator_loss = GANLoss(
            gan_mode=gan_mode,
            target_real_label=self.training_config.get('label_smooth_real', 0.9),
            target_fake_label=self.training_config.get('label_smooth_fake', 0.1),
            label_smoothing=True
        ).to(self.device)
    
    def _setup_schedulers(self):
        """Initialize learning rate schedulers."""
        scheduler_type = self.training_config.get('lr_scheduler', 'linear')
        
        if scheduler_type == 'linear':
            # Linear decay starting at specified epoch
            decay_start_epoch = int(
                self.training_config['num_epochs'] * 
                self.training_config.get('lr_decay_start_ratio', 0.5)
            )
            
            def lr_lambda(epoch):
                if epoch < decay_start_epoch:
                    return 1.0
                else:
                    progress = (epoch - decay_start_epoch) / (
                        self.training_config['num_epochs'] - decay_start_epoch
                    )
                    return max(1.0 - progress, self.training_config['min_lr'] / 
                              self.training_config['initial_lr_g'])
            
            self.scheduler_g = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer_g, lr_lambda
            )
            self.scheduler_d = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer_d, lr_lambda
            )
        
        elif scheduler_type == 'cosine':
            self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_g, T_max=self.training_config['num_epochs']
            )
            self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_d, T_max=self.training_config['num_epochs']
            )
        
        else:
            self.scheduler_g = None
            self.scheduler_d = None
    
    def _setup_early_stopping(self):
        """Initialize early stopping."""
        if self.training_config.get('patience', 0) > 0:
            save_path = Path(self.checkpoint_config.get('save_dir', 'checkpoints'))
            save_path.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = save_path / self.checkpoint_config.get(
                'filename', 'best_model.pt'
            )
            
            self.early_stopping = EarlyStopping(
                patience=self.training_config['patience'],
                verbose=True,
                path=str(checkpoint_path),
                delta=self.training_config.get('min_delta', 0.0)
            )
        else:
            self.early_stopping = None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        epoch_metrics = {
            'g_loss': [],
            'd_loss': [],
            'l1_loss': [],
            'sam_loss': []
        }
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.current_epoch + 1}/{self.training_config["num_epochs"]}',
            leave=False
        )
        
        for batch_idx, (rgb_batch, spectral_batch) in enumerate(progress_bar):
            rgb_batch = rgb_batch.to(self.device)
            spectral_batch = spectral_batch.to(self.device)
            
            # Train discriminator multiple times per generator update
            d_losses = []
            for _ in range(self.training_config.get('d_update_freq', 1)):
                d_loss = self._train_discriminator_step(rgb_batch, spectral_batch)
                d_losses.append(d_loss)
            
            # Train generator
            g_loss, g_metrics = self._train_generator_step(rgb_batch, spectral_batch)
            
            # Record metrics
            epoch_metrics['g_loss'].append(g_loss)
            epoch_metrics['d_loss'].append(np.mean(d_losses))
            epoch_metrics['l1_loss'].append(g_metrics.get('l1', 0.0))
            epoch_metrics['sam_loss'].append(g_metrics.get('sam', 0.0))
            
            # Update progress bar
            progress_bar.set_postfix({
                'G_Loss': f'{g_loss:.4f}',
                'D_Loss': f'{np.mean(d_losses):.4f}',
                'LR': f'{self.optimizer_g.param_groups[0]["lr"]:.2e}'
            })
        
        # Calculate epoch averages
        epoch_averages = {
            key: np.mean(values) for key, values in epoch_metrics.items()
        }
        
        return epoch_averages
    
    def _train_discriminator_step(self, rgb_batch: torch.Tensor, 
                                 spectral_batch: torch.Tensor) -> float:
        """Single discriminator training step."""
        self.optimizer_d.zero_grad()
        
        batch_size = rgb_batch.size(0)
        
        # Train with real samples
        real_output = self.discriminator(rgb_batch, spectral_batch)
        real_loss = self.discriminator_loss(real_output, target_is_real=True)
        
        # Train with fake samples
        with torch.no_grad():
            fake_spectral = self.generator(rgb_batch)
        
        fake_output = self.discriminator(rgb_batch, fake_spectral.detach())
        fake_loss = self.discriminator_loss(fake_output, target_is_real=False)
        
        # Total discriminator loss
        d_loss = (real_loss + fake_loss) * 0.5
        d_loss.backward()
        
        # Gradient clipping
        if self.training_config.get('grad_clip_norm', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                self.training_config['grad_clip_norm']
            )
        
        self.optimizer_d.step()
        
        return d_loss.item()
    
    def _train_generator_step(self, rgb_batch: torch.Tensor, 
                             spectral_batch: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """Single generator training step."""
        self.optimizer_g.zero_grad()
        
        # Generate fake spectral images
        fake_spectral = self.generator(rgb_batch)
        
        # Get discriminator output for adversarial loss
        fake_output = self.discriminator(rgb_batch, fake_spectral)
        
        # Calculate combined generator loss
        g_loss, loss_components = self.generator_loss(
            fake_spectral, spectral_batch, fake_output
        )
        
        g_loss.backward()
        
        # Gradient clipping
        if self.training_config.get('grad_clip_norm', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                self.training_config['grad_clip_norm']
            )
        
        self.optimizer_g.step()
        
        # Convert loss components to floats
        loss_components_float = {
            key: value.item() if isinstance(value, torch.Tensor) else value
            for key, value in loss_components.items()
        }
        
        return g_loss.item(), loss_components_float
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.generator.eval()
        self.discriminator.eval()
        
        val_metrics = {
            'g_loss': [],
            'l1_loss': [],
            'sam_loss': []
        }
        
        with torch.no_grad():
            for rgb_batch, spectral_batch in tqdm(self.val_loader, 
                                                desc='Validation', leave=False):
                rgb_batch = rgb_batch.to(self.device)
                spectral_batch = spectral_batch.to(self.device)
                
                # Generate predictions
                fake_spectral = self.generator(rgb_batch)
                
                # Calculate losses (reconstruction only for validation)
                g_loss, loss_components = self.generator_loss(
                    fake_spectral, spectral_batch, discriminator_output=None
                )
                
                val_metrics['g_loss'].append(g_loss.item())
                val_metrics['l1_loss'].append(loss_components['l1'].item())
                val_metrics['sam_loss'].append(loss_components['sam'].item())
        
        # Calculate averages
        val_averages = {
            key: np.mean(values) for key, values in val_metrics.items()
        }
        
        return val_averages
    
    def train(self) -> Dict[str, list]:
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Generator parameters: {self.generator.get_num_parameters():,}")
        self.logger.info(f"Discriminator parameters: {self.discriminator.get_num_parameters():,}")
        
        start_time = time.time()
        
        for epoch in range(self.training_config['num_epochs']):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate_epoch()
            
            # Update learning rates
            if self.scheduler_g:
                self.scheduler_g.step()
            if self.scheduler_d:
                self.scheduler_d.step()
            
            # Record history
            self.training_history['train_g_loss'].append(train_metrics['g_loss'])
            self.training_history['train_d_loss'].append(train_metrics['d_loss'])
            self.training_history['train_l1_loss'].append(train_metrics['l1_loss'])
            self.training_history['train_sam_loss'].append(train_metrics['sam_loss'])
            self.training_history['val_g_loss'].append(val_metrics['g_loss'])
            self.training_history['val_l1_loss'].append(val_metrics['l1_loss'])
            self.training_history['val_sam_loss'].append(val_metrics['sam_loss'])
            self.training_history['learning_rates'].append(
                self.optimizer_g.param_groups[0]['lr']
            )
            
            # Log progress
            elapsed_time = time.time() - start_time
            self._log_epoch_results(epoch, train_metrics, val_metrics, elapsed_time)
            
            # Early stopping check
            if self.early_stopping:
                monitor_metric = val_metrics[
                    self.checkpoint_config.get('monitor', 'g_loss').replace('val_', '')
                ]
                self.early_stopping(monitor_metric, self.generator)
                
                if self.early_stopping.early_stop:
                    self.logger.info("Early stopping triggered")
                    break
        
        # Load best model if early stopping was used
        if self.early_stopping:
            self.logger.info("Loading best model from checkpoint...")
            self.generator.load_state_dict(torch.load(self.early_stopping.path))
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        return self.training_history
    
    def _log_epoch_results(self, epoch: int, train_metrics: Dict[str, float],
                          val_metrics: Dict[str, float], elapsed_time: float):
        """Log results for current epoch."""
        current_lr = self.optimizer_g.param_groups[0]['lr']
        
        log_message = (
            f"Epoch [{epoch + 1:03d}/{self.training_config['num_epochs']:03d}] | "
            f"Time: {elapsed_time:.1f}s | "
            f"LR: {current_lr:.1e} | "
            f"Train G: {train_metrics['g_loss']:.4f} "
            f"(D: {train_metrics['d_loss']:.4f}) | "
            f"Val G: {val_metrics['g_loss']:.4f} "
            f"(L1: {val_metrics['l1_loss']:.4f}, "
            f"SAM: {val_metrics['sam_loss']:.4f})"
        )
        
        self.logger.info(log_message)
    
    def save_checkpoint(self, filepath: str, include_optimizer: bool = True):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'training_history': self.training_history,
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        
        if include_optimizer:
            checkpoint.update({
                'optimizer_g_state_dict': self.optimizer_g.state_dict(),
                'optimizer_d_state_dict': self.optimizer_d.state_dict()
            })
            
            if self.scheduler_g:
                checkpoint['scheduler_g_state_dict'] = self.scheduler_g.state_dict()
            if self.scheduler_d:
                checkpoint['scheduler_d_state_dict'] = self.scheduler_d.state_dict()
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.training_history = checkpoint['training_history']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if load_optimizer and 'optimizer_g_state_dict' in checkpoint:
            self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
            
            if self.scheduler_g and 'scheduler_g_state_dict' in checkpoint:
                self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
            if self.scheduler_d and 'scheduler_d_state_dict' in checkpoint:
                self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
    
    def save_training_history(self, filepath: str):
        """Save training history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        self.logger.info(f"Training history saved to {filepath}")
