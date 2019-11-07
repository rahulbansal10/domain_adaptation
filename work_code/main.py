import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import pdb
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split
import pickle
import random
from datasets import office_loader, feature_dataset
from feature_extractor import VGG_FeatureExtractor
from modules import content_module, style_module, reconstruction, content_classifier_module
from modules import adv_classifier_module, domain_identifier_module, class_differentiator_module
from training_protocol import training_protocol
from testing import Test

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


device = "cuda:0"
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
source_index_dict = {}
target_index_dict = {}
for i in range(31):
    source_index_dict[i] = np.where(source_labels==i)[0]
    target_index_dict[i] = np.where(target_labels==i)[0]
features = np.concatenate([source_features, target_features], axis = 0)
labels = np.concatenate([source_labels, target_labels], axis = 0)

features_dataset = feature_dataset(features, labels, device = device)
source_features_dataset = feature_dataset(source_features, source_labels, device = device)
target_features_dataset = feature_dataset(target_features, target_labels, device = device)

features_loader = DataLoader(features_dataset, batch_size = 512, shuffle = True)
source_features_loader = DataLoader(source_features_dataset, batch_size = 512, shuffle = True)
target_features_loader = DataLoader(target_features_dataset, batch_size = 512, shuffle = True)

Cs = content_module().to(device)
content_clf = content_classifier_module().to(device)
Ss = style_module().to(device)
Rs = reconstruction().to(device)
adv_clf = adv_classifier_module().to(device)

# Cs.load_state_dict(torch.load("../modules/Cs_module"))
# content_clf.load_state_dict(torch.load("../modules/content_clf_module"))

# Ss.load_state_dict(torch.load("../modules/Ss_module"))
# Rs.load_state_dict(torch.load("../modules/Rs_module"))
# adv_clf.load_state_dict(torch.load("../modules/adv_clf"))

# Ct = content_module().to(device)
# St = style_module().to(device)
# Rt = reconstruction().to(device)
# Ct.load_state_dict(Cs.state_dict())
# St.load_state_dict(Ss.state_dict())
# Rt.load_state_dict(Rs.state_dict())

CD = class_differentiator_module().to(device)
# CD.load_state_dict(torch.load("../modules/CD_module"))
ti = training_protocol(device)


# ti.train_content_module(data_loader = source_features_loader, C = Cs, content_clf= content_clf, epochs=200)
# ti.train_style_module(data_loader = source_features_loader, C = Cs, S = Ss, R = Rs, adv_clf = adv_clf, epochs=1000)
# train_class_differentiator_module(source_features_loader, CD, epochs=1000)


ti.test_content(target_features_loader, Cs, content_clf)
# ti.test_style(target_features_loader, Ss, adv_clf)
# ti.test_class_differentiator_module(data_loader = source_features_loader, CD = CD, epochs = 6000)

ti.train(source_index_dict, source_features, source_labels, target_features_loader, Cs,content_clf, Ss, Rs, Ct, St, Rt, CD, epochs=500)
# Ct.load_state_dict(torch.load("../modules/Ct_module"))
ti.test_content(target_features_loader, Ct, content_clf)