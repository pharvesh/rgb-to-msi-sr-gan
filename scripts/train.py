
"""
Training Script for RGB to Multispectral Image Super-Resolution

"""

import os
import yaml
import torch
import logging
from torch.utils.data import DataLoader
from models.generator import ResAttUNetGenerator
from models.discriminator import PatchGANDiscriminator
from utils.datasets import RGBToMSIDataset
from utils.losses import GANLoss, L1Loss, SAMLoss, PerceptualLoss
from utils.early_stopping import EarlyStopping
from utils.metrics import compute_metrics
from utils.weight_init import initialize_weights
from torchvision.utils import save_image
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import random

# ------------------------------- Utility Functions ------------------------------- #

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(config):
    if config["hardware"]["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config["hardware"]["device"])

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# ------------------------------- Main Training ------------------------------- #

def train(config):

    # Setup
    set_seed(config["seed"])
    os.makedirs(config["checkpoint"]["save_dir"], exist_ok=True)
    os.makedirs(config["logging"]["log_dir"], exist_ok=True)
    device = get_device(config)

    writer = SummaryWriter(config["logging"]["log_dir"])
    logging.basicConfig(level=logging.getLevelName(config["logging"]["log_level"]))

    # Datasets and Loaders
    train_dataset = RGBToMSIDataset(config["data"], split="train")
    val_dataset = RGBToMSIDataset(config["data"], split="val")

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"],
                              shuffle=True, num_workers=config["data"]["num_workers"],
                              pin_memory=config["data"]["pin_memory"])
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["val_batch_size"],
                            shuffle=False, num_workers=config["data"]["num_workers"],
                            pin_memory=config["data"]["pin_memory"])

    # Models
    gen = ResAttUNetGenerator(in_channels=config["model"]["input_channels"],
                              out_channels=config["model"]["output_channels"],
                              base_features=config["model"]["base_unet_features"],
                              bilinear=config["model"]["bilinear_upsampling"]).to(device)
    disc = PatchGANDiscriminator(in_channels=config["model"]["input_channels"] +
                                               config["model"]["output_channels"],
                                 features_d=config["model"]["disc_features"]).to(device)

    initialize_weights(gen)
    initialize_weights(disc)

    # Losses
    gan_loss_fn = GANLoss().to(device)
    l1_loss_fn = L1Loss().to(device)
    sam_loss_fn = SAMLoss().to(device)
    perceptual_loss_fn = PerceptualLoss().to(device)

    # Optimizers
    opt_g = Adam(gen.parameters(), lr=config["training"]["initial_lr_g"], betas=(0.5, 0.999))
    opt_d = Adam(disc.parameters(), lr=config["training"]["initial_lr_d"], betas=(0.5, 0.999))

    # Learning rate schedulers
    def lr_lambda(epoch):
        start_decay = int(config["training"]["num_epochs"] * config["training"]["lr_decay_start_ratio"])
        if epoch < start_decay:
            return 1.0
        return max(config["training"]["min_lr"] / config["training"]["initial_lr_g"],
                   1 - (epoch - start_decay) / (config["training"]["num_epochs"] - start_decay))

    scheduler_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda)
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda)

    # Early Stopping
    early_stopper = EarlyStopping(patience=config["training"]["patience"],
                                  mode=config["checkpoint"]["mode"],
                                  monitor=config["checkpoint"]["monitor"],
                                  save_path=os.path.join(config["checkpoint"]["save_dir"],
                                                         config["checkpoint"]["filename"]))

    # Training Loop
    for epoch in range(config["training"]["num_epochs"]):
        gen.train()
        disc.train()

        epoch_g_loss = 0
        epoch_d_loss = 0

        for i, (rgb, ms) in enumerate(tqdm(train_loader)):
            rgb, ms = rgb.to(device), ms.to(device)

            #######################
            # Train Discriminator #
            #######################
            for _ in range(config["training"]["d_update_freq"]):
                fake_ms = gen(rgb).detach()
                d_real = disc(torch.cat([rgb, ms], dim=1))
                d_fake = disc(torch.cat([rgb, fake_ms], dim=1))
                d_loss = gan_loss_fn(d_real, True, config["training"]["label_smooth_real"]) + \
                         gan_loss_fn(d_fake, False, config["training"]["label_smooth_fake"])

                opt_d.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), config["training"]["grad_clip_norm"])
                opt_d.step()

            ###################
            # Train Generator #
            ###################
            fake_ms = gen(rgb)
            g_fake = disc(torch.cat([rgb, fake_ms], dim=1))

            l1 = l1_loss_fn(fake_ms, ms)
            sam = sam_loss_fn(fake_ms, ms)
            adv = gan_loss_fn(g_fake, True)
            perc = perceptual_loss_fn(fake_ms, ms)

            g_loss = config["training"]["lambda_l1"] * l1 + \
                     config["training"]["lambda_sam"] * sam + \
                     config["training"]["lambda_adv"] * adv + \
                     perc

            opt_g.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), config["training"]["grad_clip_norm"])
            opt_g.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        scheduler_g.step()
        scheduler_d.step()

        # Validation
        gen.eval()
        val_losses = []
        with torch.no_grad():
            for rgb, ms in val_loader:
                rgb, ms = rgb.to(device), ms.to(device)
                fake_ms = gen(rgb)
                l1 = l1_loss_fn(fake_ms, ms)
                val_losses.append(l1.item())

        avg_val_loss = np.mean(val_losses)
        writer.add_scalar("Loss/train_generator", epoch_g_loss / len(train_loader), epoch)
        writer.add_scalar("Loss/train_discriminator", epoch_d_loss / len(train_loader), epoch)
        writer.add_scalar("Loss/val_reconstruction", avg_val_loss, epoch)

        logging.info(f"Epoch {epoch} | G Loss: {epoch_g_loss:.4f} | D Loss: {epoch_d_loss:.4f} | Val L1: {avg_val_loss:.4f}")

        early_stopper.step(avg_val_loss, gen)
        if early_stopper.early_stop:
            logging.info("Early stopping triggered.")
            break

    writer.close()
    logging.info("Training Complete.")


if __name__ == "__main__":
    config = load_config("config/config.yaml")
    train(config)
