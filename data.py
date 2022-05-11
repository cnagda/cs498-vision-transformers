from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
import pytorch_lightning as pl


class CIFAR10DataModule(pl.LightningDataModule):
    """
    Lightning module for CIFAR10.
    """
    def __init__(self, data_dir, batch_size=128, num_workers=2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        """
        Download data if not in dir.
        """
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage):
        """
        Augment data and store splits as datasets.
        """
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=(32, 32), ratio=(0.9, 1,1)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # assign splits for use in dataloaders
        self.full_dataset = CIFAR10(root=self.data_dir, train=True, 
                                    transform=train_transform)
        self.train_dataset, self.val_dataset = data.random_split(
            self.full_dataset, lengths=[45000, 5000])
        self.test_dataset = CIFAR10(root=self.data_dir, train=False, 
                                    transform=test_transform)

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset,
                               batch_size=self.batch_size,
                               shuffle=True,
                               num_workers=self.num_workers)

    def val_dataloader(self):
        return data.DataLoader(self.test_dataset,
                               batch_size=self.batch_size,
                               shuffle=True,
                               num_workers=self.num_workers)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset,
                               batch_size=self.batch_size,
                               shuffle=True,
                               num_workers=self.num_workers)
                               
# TODO: module for TinyImageNet