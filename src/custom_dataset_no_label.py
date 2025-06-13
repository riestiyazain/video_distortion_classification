import os
import torch
from PIL import Image
from torchvision import datasets, transforms
from PIL import Image, UnidentifiedImageError

class ImageFolderWithoutLabels(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = [
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
            if not fname.startswith('.') and fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except UnidentifiedImageError:
            # Skip invalid images
            raise ValueError(f"Cannot identify image file {image_path}")
        if self.transform:
            image = self.transform(image)
        return image, image_path  # Return image and its path