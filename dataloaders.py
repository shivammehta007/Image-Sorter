import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from config import TRAIN_DATA_FOLDER, TEST_DATA_FOLDER

def get_data_loaders(traintransform, testtransform):
    train_dataset = datasets.ImageFolder(TRAIN_DATA_FOLDER, transform=traintransform)
    test_dataset = datasets.ImageFolder(TEST_DATA_FOLDER, transform=testtransform)

    trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    return trainloader, testloader
