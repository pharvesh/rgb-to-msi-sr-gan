
"""
Training Script for RGB to Multispectral Image Super-Resolution

This script handles the complete training pipeline including model initialization,
data loading, training loop, and checkpointing.

Usage:
    python scripts/train.py --config configs/default_config.yaml
    python scripts/train.py --config configs/custom_config.yaml --resume checkpoints/latest.pt

"""

import argparse
import logging
import os
import sys
import torch
import yaml
from pathlib import Path
import numpy as np
from datetime import datetime

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.generator import ResAttentionUNet
from src.models.discriminator import PatchGANDiscriminator
from src.data.dataset import create_dataloaders
from src.utils.config import load_config, validate_config
from src.utils.logging_utils import setup_logging
from scripts.trainer import GANTrainer
from scripts.utils import set_seed, save_config, create_checkpoint_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train RGB to Multispectral Image Super-Resolution GAN'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--resume', 
        type=str, 
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='experiments',
        help='Base directory for experiment outputs'
    )
    
    parser.add_argument(
        '--experiment-name', 
        type=str, 
        default=None,
        help='Name for this experiment (default: auto-generated)'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode with additional logging'
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Perform a dry run without actual training'
    )
    
    return parser.parse_args()


def setup_experiment_dir(base_dir: str, experiment_name: str = None) -> Path:
    """Create experiment directory structure."""
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"rgb2ms_gan_{timestamp}"
    
    exp_dir = Path(base_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'config').mkdir(exist_ok=True)
    (exp_dir / 'results').mkdir(exist_ok=True)
    
    return exp_dir


def initialize_models(config: dict, device: torch.device) -> tuple:
    """Initialize generator and discriminator models."""
    model_config = config['model']
    
    # Initialize generator
    generator = ResAttentionUNet(
        n_channels=model_config['input_channels'],
        n_classes=model_config['output_channels'],
        base_features=model_config['base_unet_features'],
        bilinear=model_config.get('bilinear_upsampling', True)
    ).to(device)
    
    # Initialize discriminator
    discriminator = PatchGANDiscriminator(
        in_channels=model_config['input_channels'] + model_config['output_channels'],
        features_d=model_config['disc_features']
    ).to(device)
    
    # Initialize weights
    generator.initialize_weights()
    discriminator.initialize_weights()
    
    return generator, discriminator


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load and validate configuration
    config = load_config(args.config)
    validate_config(config)
    
    # Set random seed for reproducibility
    if 'seed' in config:
        set_seed(config['seed'])
    
    # Setup experiment directory
    exp_dir = setup_experiment_dir(args.output_dir, args.experiment_name)
    
    # Setup logging
    log_file = exp_dir / 'logs' / 'training.log'
    logger = setup_logging(log_file, debug=args.debug)
    
    logger.info("="*60)
    logger.info("RGB to Multispectral Image Super-Resolution Training")
    logger.info("="*60)
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Configuration file: {args.config}")
    
    # Save configuration to experiment directory
    config_save_path = exp_dir / 'config' / 'config.yaml'
    save_config(config, config_save_path)
    
    # Determine device
    if config.get('hardware', {}).get('device') == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config.get('hardware', {}).get('device', 'cpu'))
    
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader = create_dataloaders(config)
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Training batches per epoch: {len(train_loader)}")
        
        # Initialize models
        logger.info("Initializing models...")
        generator, discriminator = initialize_models(config, device)
        
        logger.info(f"Generator parameters: {generator.get_num_parameters():,}")
        logger.info(f"Discriminator parameters: {discriminator.get_num_parameters():,}")
        
        # Update config with experiment directory paths
        config['checkpoint']['save_dir'] = str(exp_dir / 'checkpoints')
        config['logging']['log_dir'] = str(exp_dir / 'logs')
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            logger=logger
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming training from {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Dry run check
        if args.dry_run:
            logger.info("Dry run mode - skipping actual training")
            logger.info("Configuration and model initialization successful")
            return
        
        # Start training
        logger.info("Starting training...")
        training_history = trainer.train()
        
        # Save final results
        logger.info("Saving final results...")
        
        # Save final model
        final_model_path = exp_dir / 'checkpoints' / 'final_model.pt'
        torch.save({
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'config': config,
            'training_history': training_history
        }, final_model_path)
        
        # Save training history
        history_path = exp_dir / 'results' / 'training_history.json'
        trainer.save_training_history(history_path)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Final model saved to: {final_model_path}")
        logger.info(f"Training history saved to: {history_path}")
        logger.info(f"All outputs saved to: {exp_dir}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
        # Save current state
        if 'trainer' in locals():
            interrupt_path = exp_dir / 'checkpoints' / 'interrupted.pt'
            trainer.save_checkpoint(interrupt_path)
            logger.info(f"Current state saved to: {interrupt_path}")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == '__main__':
    main()
