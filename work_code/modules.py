import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import pdb

class content_module(nn.Module):
    def __init__(self, feat_dim=4096, content_dim=31):
        super(content_module, self).__init__()
        self.feat_dim = feat_dim
        self.content_dim = content_dim
        self.fc1 = nn.Linear(self.feat_dim,int(self.feat_dim/4))
        self.fc2 = nn.Linear(int(self.feat_dim/4), self.content_dim)
    
    def forward(self, x):
        h_1 = (F.relu(self.fc1(x)))
        content = F.relu(self.fc2(h_1))
        return content



class style_module(nn.Module):
    def __init__(self, feat_dim=4096, style_dim = 100):
        super(style_module, self).__init__()
        self.feat_dim = feat_dim
        self.style_dim = style_dim
        self.fc1 = nn.Linear(self.feat_dim, int(self.feat_dim/4))
        self.fc2 = nn.Linear(int(self.feat_dim/4), self.style_dim)
    
    def forward(self, x):
        h_1 = F.relu(self.fc1(x))
        style = F.relu(self.fc2(h_1))
        return style

class reconstruction(nn.Module):
    def __init__(self, feat_dim=4096, content_dim = 31, style_dim = 100):
        super(reconstruction, self).__init__()
        self.feat_dim = feat_dim
        self.style_dim = style_dim
        self.content_dim = content_dim
        self.input_dim = style_dim + content_dim
        self.fc1 = nn.Linear(self.input_dim, int(self.feat_dim/4))
        self.fc2 = nn.Linear(int(self.feat_dim/4), self.feat_dim)
    
    def forward(self, content, style):
        m = nn.Softmax(dim=1)
        x = torch.cat((content, style), 1)
        h_1 = F.relu(self.fc1(x))
        h_2 = F.relu(self.fc2(h_1))
        return h_2

class adv_classifier_module(nn.Module):
    def __init__(self, num_classes=31, style_dim = 100):
        super(adv_classifier_module, self).__init__()
        self.style_dim = style_dim
        self.num_classes = num_classes
        self.fc1 = nn.Linear(self.style_dim, self.num_classes)
    
    def forward(self, x):
        logits = F.relu(self.fc1(x))
        return logits

class domain_identifier_module(nn.Module):
    def __init__(self, feat_dim = 4096):
        super(domain_identifier_module, self).__init__()
        self.feat_dim = feat_dim
        self.fc1 = nn.Linear(self.feat_dim, int(self.feat_dim/4))
        self.fc2 = nn.Linear(int(self.feat_dim/4), int(self.feat_dim/16))
        self.fc3 = nn.Linear(int(self.feat_dim/16), 2)
    
    def forward(self, x):
        h_1 = F.relu(self.fc1(x))
        h_2 = F.relu(self.fc2(h_1))
        h_3 = F.relu(self.fc3(h_2))
        return h_3

class class_differentiator_module(nn.Module):
    def __init__(self, feat_dim = 4096):
        super(class_differentiator_module, self).__init__()
        self.feat_dim = feat_dim
        self.fc1 = nn.Linear(int(2*self.feat_dim), int(self.feat_dim/4))
        self.fc2 = nn.Linear(int(self.feat_dim/4), int(self.feat_dim/16))
        self.fc3 = nn.Linear(int(self.feat_dim/16), 2)        
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = (F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x