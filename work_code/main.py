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
from sklearn.model_selection import train_test_split
import pickle
import random
from datasets import office_loader, feature_dataset
from feature_extractor import VGG_FeatureExtractor
from modules import content_module, style_module, reconstruction 
from modules import adv_classifier_module, domain_identifier_module, class_differentiator_module
from training_protocol import training_protocol


def read_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    f.close()
    return data

class ResizeImage():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))
    
def transform_data(resize_size=224, crop_size=224):
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  return  transforms.Compose([
        ResizeImage(resize_size),
        transforms.ToTensor(),
        normalize
    ])

def get_features(G, data_loader):  # G is the feature extractor
    X = np.zeros((1,4096))
    Y = np.zeros((1), dtype= int)
    for images, labels in data_loader:
        features = G(images).data.numpy()
        labels = labels.numpy()
        X = np.concatenate((X,features), axis=0)
        Y = np.concatenate((Y,labels), axis=0)
    return X[1:], Y[1:]


# transform = transform_data()
# source = office_loader(name = "office/amazon_31_list.txt", transform = transform)
# target = office_loader(name = "office/webcam_31_list.txt", transform = transform)
# source_loader = DataLoader(source, batch_size = 64,shuffle = True)
# target_loader = DataLoader(target, batch_size = 64,shuffle = True)
# G = VGG_FeatureExtractor()
# source_features, source_labels = get_features(G, source_loader)
# target_features, target_labels = get_features(G, target_loader)

source_features, source_labels = read_data("../features/source_features.pkl"), read_data("../features/source_labels.pkl")
target_features, target_labels = read_data("../features/target_features.pkl"), read_data("../features/target_labels.pkl")

features = np.concatenate([source_features, target_features], axis = 0)
labels = np.concatenate([source_labels, target_labels], axis = 0)

features_dataset = feature_dataset(features, labels)
source_features_dataset = feature_dataset(source_features, source_labels)
target_features_dataset = feature_dataset(target_features, target_labels)

features_loader = DataLoader(features_dataset, batch_size = 512, shuffle = True)
source_features_loader = DataLoader(source_features_dataset, batch_size = 512, shuffle = True)
target_features_loader = DataLoader(target_features_dataset, batch_size = 512, shuffle = True)

Cs = content_module()
Ss = style_module()
Rs = reconstruction()
adv_clf = adv_classifier_module()
Cs.load_state_dict(torch.load("../modules/Cs_module"))
Ss.load_state_dict(torch.load("../modules/Ss_module"))
Rs.load_state_dict(torch.load("../modules/Rs_module"))

Ct = content_module()
St = style_module()
Rt = reconstruction()

CD = class_differentiator_module()
ti = training_protocol()

# ti.train_content_module(data_loader = source_features_loader, C = Cs, epochs=100)
# ti.train_style_module(data_loader = features_loader, C = Cs, S = Ss, R = Rs, adv_clf = adv_clf, epochs=100)
ti.train_class_differentiator_module(data_loader = source_features_loader, CD = CD, epochs = 500)