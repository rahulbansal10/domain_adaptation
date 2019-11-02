import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
from torchvision import models

class VGG_FeatureExtractor(nn.Module):
  def __init__(self):
    super(VGG_FeatureExtractor, self).__init__()
    model_vgg = models.vgg16(pretrained=True)
    self.features = model_vgg.features
    self.avgpool = model_vgg.avgpool
    self.classifier = model_vgg.classifier[0:3]

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), 25088)
    x = self.classifier(x)
    return x