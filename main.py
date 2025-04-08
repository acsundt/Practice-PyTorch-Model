import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from PlayingCardDataset import PlayingCardDataset


def main():

    dataset_path = r"C:\Users\aidan\OneDrive\Documents\ML_Stuff\Card_Images_Dataset\train"

    dataset = PlayingCardDataset(data_dir=dataset_path)

    print("length of dataset = ", len(dataset))

    target_to_class = {v: k for k, v in ImageFolder(dataset_path).class_to_idx.items()}
    # print(target_to_class)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = PlayingCardDataset(dataset_path, transform)
    print(dataset[100])






main()