import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms, models

from PIL import Image



def load_transform_data(data_dir):
    '''Load data from ImageFolder and apply transformations
       on data
    '''

    # Means and std calculated from ImageNet images
    means = [0.485, 0.456, 0.406]
    stdv = [0.229, 0.224, 0.225]

    # Training transforms
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(means, stdv)])

    # Testing and validation transforms
    test_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(means, stdv)])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    validation_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)

    # Define dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True)

    return trainloader, testloader, validationloader, train_data, test_data, validation_data



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns PyTorch tensor
    '''

    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    im = Image.open(image)

    # Resize, Crop and Normalize the image
    transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(means, std)])
    im_tensor = transform(im)

    return im_tensor
