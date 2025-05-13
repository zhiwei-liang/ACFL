import os
import torch
from data.meta_dataset import MetaDataset, GetDataLoaderDict
from configs.default import officehome_path
from torchvision import transforms
import random
officehome_name_dict = {
    'p': 'Product',
    'a': 'Art',
    'c': 'Clipart',
    'r': 'Real_World',
}
transform_train = transforms.Compose(
            [transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

transform_test = transforms.Compose(
            [transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


class OfficeHome_SingleDomain():
    def __init__(self, root_path=officehome_path, domain_name='p', split='train', train_transform=None, seed=0):
        self.domain_name = domain_name
        assert domain_name in officehome_name_dict.keys(), 'domain_name must be in {}'.format(officehome_name_dict.keys())
        self.root_path = root_path
        self.domain = officehome_name_dict[domain_name]
        self.domain_label = list(officehome_name_dict.keys()).index(domain_name)
        self.txt_path = os.path.join(root_path, '{}_img_label_list.txt'.format(self.domain))
        
        self.split = split
        assert self.split in ['train', 'val', 'test'] , 'split must be train, val or test'
        
        if train_transform is not None:
            self.transform = train_transform
        else:
            self.transform = transform_test
        self.seed = seed
        
        self.imgs, self.labels = OfficeHome_SingleDomain.read_txt(self.txt_path, self.root_path)
        
        if self.split == 'train' or self.split == 'val':
            random.seed(self.seed)
            train_img, val_img = OfficeHome_SingleDomain.split_list(self.imgs, 0.9)
            random.seed(self.seed)
            train_label, val_label = OfficeHome_SingleDomain.split_list(self.labels, 0.9)
            if self.split == 'train':
                self.imgs, self.labels = train_img, train_label
            elif self.split == 'val':
                self.imgs, self.labels = val_img, val_label
                
        self.dataset = MetaDataset(self.imgs, self.labels, self.domain_label, self.transform) # get数据集
    
    @staticmethod
    def split_list(l, ratio):
        assert ratio > 0 and ratio < 1
        random.shuffle(l) # 打乱list
        train_size = int(len(l)*ratio)
        train_l = l[:train_size]
        val_l = l[train_size:]
        return train_l, val_l
        
    @staticmethod
    def read_txt(txt_path, root_path):
        imgs = []
        labels = []
        with open(txt_path, 'r') as f:
            contents = f.readlines()
            
        for line_txt in contents:
            line_txt = line_txt.replace('\n', '')
            line_txt_list = line_txt.split(' ')
            if len(line_txt_list) != 3:
                imgs.append(os.path.join(root_path, line_txt_list[0]))
                labels.append(int(line_txt_list[1]))
            else:
                imgs_path = line_txt_list[0] + " " + line_txt_list[1]
                imgs.append(os.path.join(root_path, imgs_path))
                labels.append(int(line_txt_list[-1]))
            
        return imgs, labels

class OfficeHome_FedDG():
    def __init__(self, test_domain='p', batch_size=16, seed=0):
        self.batch_size = batch_size
        self.domain_list = list(officehome_name_dict.keys())
        # self.test_domain = test_domain
        self.train_domain_list = self.domain_list.copy()
        # self.train_domain_list.remove(self.test_domain)  
        self.seed = seed
        
        self.site_dataset_dict = {}
        self.site_dataloader_dict = {}
        for domain_name in self.domain_list:
            self.site_dataloader_dict[domain_name], self.site_dataset_dict[domain_name] = OfficeHome_FedDG.SingleSite(domain_name, self.batch_size, self.seed)
            
        
        # self.test_dataset = self.site_dataset_dict[self.test_domain]['test']
        # self.test_dataloader = self.site_dataloader_dict[self.test_domain]['test']
        
          
    @staticmethod
    def SingleSite(domain_name, batch_size=16, seed=0):
        dataset_dict = {
            'train': OfficeHome_SingleDomain(domain_name=domain_name, split='train', train_transform=transform_train, seed=seed).dataset,
            'val': OfficeHome_SingleDomain(domain_name=domain_name, split='val', seed=seed).dataset,
            'test': OfficeHome_SingleDomain(domain_name=domain_name, split='test', seed=seed).dataset,
        }
        dataloader_dict = GetDataLoaderDict(dataset_dict, batch_size)
        return dataloader_dict, dataset_dict
        
    def GetData(self):
        return self.site_dataloader_dict, self.site_dataset_dict

def GenFileList():
    '''
    get path label list on all domains
    '''
    total_class_list = None
    for domain_name in officehome_name_dict.keys():
        domain = officehome_name_dict[domain_name]
        domain_path = os.path.join(officehome_path, domain)
        class_list = os.listdir(domain_path)
        class_list.sort()
        if total_class_list is None:
            total_class_list = class_list
        
        assert total_class_list == class_list, 'class_list must be same'
    
    domain_file_dict = {}
    for domain_name in officehome_name_dict.keys():
        domain_file_dict[domain_name] = []
        domain = officehome_name_dict[domain_name]
        domain_path = os.path.join(officehome_path, domain)
        for label_idx, class_name in enumerate(total_class_list):
            class_path = os.path.join(domain_path, class_name)
            file_list = os.listdir(class_path)
            
            for file_name in file_list:
                file_path = os.path.join(class_path, file_name)
                domain_file_dict[domain_name].append(file_path + ' ' + str(label_idx) + '\n')
        
    return domain_file_dict







