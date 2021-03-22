import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

# class for Heatmap dataset
class HeatmapDataset(Dataset):
    """heatmap dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the folders. Each folder refers to a class.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.num_classes = len(os.listdir(self.root_dir))
        self.images = []
        self.targets = []
        if "Images" in os.listdir(self.root_dir):
            images_path = sorted(os.listdir(os.path.join(self.root_dir, "Images")))
            self.images.extend([os.path.join(self.root_dir, "Images", image) for image in images_path if any(('.jpg' in image, 
                                                                                                              '.png' in image))])
        if "Annotations" in os.listdir(self.root_dir):
            ann_file = os.path.join(self.root_dir, "Annotations", "annotation.txt")
            with open(ann_file, "rb") as f:
                target_lines = f.readlines()
            for line in target_lines:
                self.targets.append(line.split()[1])
#             self.targets.extend([int(f)] * len(images_path))
        
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        target = int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        return img, target
