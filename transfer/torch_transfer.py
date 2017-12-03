import torch
import torch.nn as nn
import torch optim as optim
from torch.autograd import Varialble
from torchvision import transforms, models, datasets

import os
import time
import matplotlib.pyplot as plt
import numpy as np

data_transforms = {
        'train': transforms.Compose([transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'val':transforms.Compose([transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.Totensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
}

data_dir = '../../data/hymenoptera_data'
image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir, x),
                                         data_transforms[x]) for x in ['train','val']}
dataloaders = {x:len(image_datasets[x]) for x in ['train', 'val']}
