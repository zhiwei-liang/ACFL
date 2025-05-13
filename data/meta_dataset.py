import torch
from torch.utils.data import Dataset
from PIL import Image

import numpy as np
import cv2

dataloader_kwargs = {'num_workers': 4, 'pin_memory': True}
def GetDataLoaderDict(dataset_dict, batch_size):
    dataloader_dict = {}
    for dataset_name in dataset_dict.keys():
        if 'train' in dataset_name:
            dataloader_dict[dataset_name] = torch.utils.data.DataLoader(dataset_dict[dataset_name], batch_size=batch_size, shuffle=True, drop_last=True, **dataloader_kwargs)
        else:
            dataloader_dict[dataset_name] = torch.utils.data.DataLoader(dataset_dict[dataset_name], batch_size=batch_size, shuffle=False, drop_last=False, **dataloader_kwargs)

    return dataloader_dict


class MetaDataset(Dataset):
    '''
    For RGB data, single client
    '''
    def __init__(self, imgs, labels, domain_label, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.domain_label = domain_label
        self.transform = transform
    
    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_class_label = self.labels[index]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, img_class_label, self.domain_label

    def __len__(self):
        return len(self.imgs)
    