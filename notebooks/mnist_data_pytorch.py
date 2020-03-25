"""
Load MNIST dataset from MNSIT and give a dictionary with it's training/validation dataloaders
"""
import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

batch_size = 100

data_train = MNIST("./data/mnist",
                   download=True,
                   train=True,
                   transform=transforms.Compose([
                       transforms.Resize((28, 28)),
                       transforms.ToTensor(),
                       #transforms.Normalize((0.1307,), (0.3081,)),
                   ]))

data_val = MNIST("./data/mnist",
                 train=False,
                 download=True,
                 transform=transforms.Compose([
                     transforms.Resize((28, 28)),
                     transforms.ToTensor(),
                     #transforms.Normalize((0.1307,), (0.3081,)),
                 ]))

dataloader_train = DataLoader(
    data_train, batch_size=batch_size, shuffle=True, num_workers=8)
dataloader_val = DataLoader(data_val, batch_size=batch_size, num_workers=8)

dataloaders = {
    "train": dataloader_train,
    "val": dataloader_val,
}