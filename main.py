import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import torch_directml

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from PlayingCardDataset import PlayingCardDataset
from SimpleCardClassifier import SimpleCardClassifier

from tqdm.notebook import tqdm
from glob import glob

from Visualizations import preprocess_image, predict, visualize_predictions


def main():

    model = SimpleCardClassifier(num_classes=53)

    device = torch_directml.device()
    print("Using device:", device)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_path = r"C:\Users\Aidan Sundt\PycharmProjects\Practice-PyTorch-Model\Card_Images_Dataset\train"
    valid_path = r"C:\Users\Aidan Sundt\PycharmProjects\Practice-PyTorch-Model\Card_Images_Dataset\valid"
    test_path = r"C:\Users\Aidan Sundt\PycharmProjects\Practice-PyTorch-Model\Card_Images_Dataset\test"

    train_dataset = PlayingCardDataset(data_dir=train_path, transform=transform)
    val_dataset = PlayingCardDataset(valid_path, transform=transform)
    test_dataset = PlayingCardDataset(test_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_epochs = 5
    train_losses, val_losses = [], []

    # class_names = test_dataset.classes

    for epoch in range(num_epochs):
        # Set model to train
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc='Training loop'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation Loop'):
                # Move inputs and labels to the device
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        # Print epoch stats
        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_losses}, Validation loss: {val_losses}")


    # Visualize
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title("Loss Over Epochs")
    plt.show()

    test_images = glob(r"C:\Users\Aidan Sundt\PycharmProjects\Practice-PyTorch-Model\Card_Images_Dataset\test/*/*")
    test_examples = np.random.choice(test_images,10)

    for example in test_examples:
        original_image, image_tensor = preprocess_image(example, transform)
        probabilities = predict(model, image_tensor, device)

        # Assuming dataset.classes gives the class names
        class_names = test_dataset.classes  # You might get this from your dataset
        visualize_predictions(original_image, probabilities, class_names)


    # print("length of dataset = ", len(dataset))

   #  target_to_class = {v: k for k, v in ImageFolder(dataset_path).class_to_idx.items()}
    # print(target_to_class)


    # dataset = PlayingCardDataset(dataset_path, transform)
    # print(dataset[100])

    # iterate over dataset
    # for image, label in dataset:
      #  break

    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # model = SimpleCardClassifier(num_classes=53)
    # print("Model: ",model)








main()