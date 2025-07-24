# RGB to Multispectral Image Super-Resolution using GAN
This repository implements a deep learning approach for spectral super-resolution, transforming 3-channel RGB images into 31-channel hyperspectral images. The model is trained on the NTIRE 2020 Spectral Recovery Challenge dataset.


## Architecture

- **Generator**: Residual Attention U-Net with skip connections and attention gates
- **Discriminator**: PatchGAN discriminator for realistic spectral generation
- **Loss Functions**: 
  - Adversarial loss for realistic generation
  - L1 reconstruction loss for pixel-wise accuracy
  - Spectral Angle Mapper (SAM) loss for spectral fidelity

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
opencv-python>=4.5.0
scipy>=1.7.0
matplotlib>=3.4.0
Pillow>=8.3.0
scikit-learn>=0.24.0
tqdm>=4.62.0
```
