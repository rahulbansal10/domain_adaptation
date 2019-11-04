import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class office_loader(Dataset):
    def __init__(self, root= "/home/iacvr/dataset/", name = "office/amazon_10_list.txt", transform=None):
        self.path = root + name
        with open(self.path,"r") as file:
            self.data = [(Image.open(path.split()[0]).convert("RGB"), int(path.split()[1])) for path in file.readlines()]
        self.transform = transform
        file.close()
    
    def __getitem__(self, idx):
        img, label = self.data[idx][0], self.data[idx][1]
        if self.transform:
            img = self.transform(img)
        else:
            img = np.asarray(img).transpose((2, 0, 1))
            img = torch.tensor(np.asarray(img))
        return img, label
    
    def __len__(self):
        return len(self.data)


class feature_dataset(Dataset):
    def __init__(self, features, labels, transform=None, device = "cpu"):
        self.features = features
        self.labels = labels
        self.device = device
            
    def __getitem__(self,idx):
        feature, label = self.features[idx], self.labels[idx]
        return torch.tensor(feature).to(self.device), torch.tensor(label).to(self.device)
    
    def __len__(self):
        return len(self.features)


