import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import optim
import torchvision.transforms as transforms
import torchvision
from fastprogress import master_bar, progress_bar
from PIL import Image
from predata import test_data,val_data
import torch
import torch.nn as nn


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ChestXrayDataset(Dataset):
    def __init__(self, folder_dir, dataframe, normalization):
        self.image_paths = [] 
        self.image_labels = [] 
        image_transformation = [
            transforms.ToTensor()
        ]
        if normalization:
            # Normalization with mean and std from ImageNet
            image_transformation.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        self.image_transformation = transforms.Compose(image_transformation)
        
        # Get all image paths and image labels from dataframe
        for index,row in dataframe.iterrows():
            path=row['Path']
            print(path)
            start_index = path.find('patient')
            #components = path.split('/')
            #patient_index = components.index('patient')
            patient_path = path[start_index:]
            image_path = os.path.join(folder_dir,patient_path)
            print(image_path)
            self.image_paths.append(image_path)
            label = row['Pleural Effusion']
            self.image_labels.append(label)

    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        
        image_path = self.image_paths[index]
        image_data = Image.open(image_path).convert("RGB") 
        image_data = self.image_transformation(image_data)
        
        return image_data, torch.tensor(self.image_labels[index])

train_dataset = ChestXrayDataset("/Users/jmac/Desktop/mip/CheXpert/test", test_data, True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, pin_memory=True)
test_dataset=ChestXrayDataset("/Users/jmac/Desktop/mip/CheXpert/val", val_data, True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, pin_memory=True)

for data, label in train_dataloader:
    print(data.size())
    print(label.size())
    break
#/Users/jmac/Desktop/mip/CheXpert/val
#/Users/jmac/Desktop/mip/CheXpert/val/patient64546/study1/view1_frontal.jpg
#/Users/jmac/Desktop/mip/CheXpert/val/patient64544