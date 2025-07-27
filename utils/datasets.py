import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import scipy.io as sio
import numpy as np


class RGBToMSIDataset(Dataset):
    def __init__(self, root_dir, rgb_subdir, spectral_subdir,
                 rgb_extension=".png", spectral_extension=".mat",
                 normalize_range=(-1, 1), transform=None):
        """
        Args:
            root_dir (str): Base folder containing RGB and spectral folders.
            rgb_subdir (str): Subfolder for RGB images.
            spectral_subdir (str): Subfolder for spectral images (.mat).
            rgb_extension (str): Extension of RGB images.
            spectral_extension (str): Extension of spectral files.
            normalize_range (tuple): Min-max normalization range for RGB.
            transform: Optional additional transforms.
        """
        self.rgb_dir = os.path.join(root_dir, rgb_subdir)
        self.spectral_dir = os.path.join(root_dir, spectral_subdir)
        self.rgb_extension = rgb_extension
        self.spectral_extension = spectral_extension
        self.normalize_range = normalize_range
        self.transform = transform

        self.filenames = [
            fname.replace(self.rgb_extension, '')
            for fname in os.listdir(self.rgb_dir)
            if fname.endswith(self.rgb_extension)
        ]
        self.filenames.sort()

        # RGB normalization
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5] * 3, std=[0.5] * 3
            ) if normalize_range == (-1, 1) else transforms.Lambda(lambda x: x)
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # Load RGB image
        rgb_path = os.path.join(self.rgb_dir, fname + self.rgb_extension)
        rgb_img = Image.open(rgb_path).convert("RGB")
        rgb_tensor = self.rgb_transform(rgb_img)

        # Load spectral image (.mat)
        spectral_path = os.path.join(self.spectral_dir, fname + self.spectral_extension)
        mat_contents = sio.loadmat(spectral_path)
        # Assuming the spectral key is the only variable in the .mat file
        spectral_data = next(iter(mat_contents.values()))
        spectral_array = np.array(spectral_data, dtype=np.float32)

        # Normalize spectral to [0, 1]
        spectral_array = spectral_array / 65535.0 if spectral_array.max() > 1 else spectral_array
        spectral_tensor = torch.from_numpy(spectral_array).permute(2, 0, 1)

        if self.transform:
            rgb_tensor = self.transform(rgb_tensor)
            spectral_tensor = self.transform(spectral_tensor)

        return rgb_tensor, spectral_tensor
