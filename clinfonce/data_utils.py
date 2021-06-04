import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import pickle
from PIL import Image
import os
import pandas as pd



class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class MyTransform:
    """Class for costomize transform"""
    def __init__(self, opt):
        super(MyTransform).__init__()
        # normolize
        if opt.dataset in ['ut-zap50k', 'ut-zap50k-sub']:
            self.mean = (0.8342, 0.8142, 0.8081)
            self.std = (0.2804, 0.3014, 0.3072)
        elif opt.dataset == 'CUB':
            self.mean = (0.4863, 0.4999, 0.4312)
            self.std = (0.2070, 0.2018, 0.2428)
        elif opt.dataset == 'Wider':
            self.mean = (0.4772, 0.4405, 0.4100)
            self.std = (0.2960, 0.2876, 0.2935)
        elif opt.dataset in ['imagenet100']:
            self.mean = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)
        else:
            raise ValueError('dataset not supported: {}'.format(opt.dataset))
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.opt = opt

    def train_transform(self, ssl=True, rrc_scale=(0.4, 1.)):
        """Transform for train"""
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=self.opt.img_size, scale=rrc_scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            self.normalize,
        ])

        if ssl:
            train_transform = TwoCropTransform(train_transform)

        return train_transform

    def val_transform(self, rrc_scale=(0.8, 1.)):
        """Transform for val"""
        resize_to = int(2 ** (np.ceil(np.log2(self.opt.img_size))))
        val_transform = transforms.Compose([
            transforms.Resize(resize_to),
            transforms.CenterCrop(size=self.opt.img_size),
            transforms.ToTensor(),
            self.normalize,
        ])

        return val_transform



class DynamicLabelDataset(Dataset):
    '''
    torch dataset
    '''
    def __init__(self, df, data_path, gran_lvl, transform=None):
        
        super(DynamicLabelDataset, self).__init__()
        self.data_path = data_path
        self.gran_lvl = gran_lvl
       
        # assign latent class column name
        if self.gran_lvl == '0':
            self.gran_lvl_label_name = 'class'
        elif self.gran_lvl == '-1':
            self.gran_lvl_label_name = 'path'
        else:
            self.gran_lvl_label_name = 'label_gran_{}'.format(gran_lvl)
        
        if -1 in df.index:
            df = df.drop([-1])
        if '-1' in df.index:
            df = df.drop(['-1'])
        print("drop -1 row")
        print(df)   

        self.df = df
        self.transform = transform

        ## processing latent class to continuous unique mapping
        unique_class = np.unique(self.df[self.gran_lvl_label_name].to_numpy())
        self.mapping = {unique_class[i]:i for i in range(unique_class.shape[0])}
        self.num_class = unique_class.shape[0]

        # clean up df
        new_df = {}
        print(f"gran_lvl_label_name {self.gran_lvl_label_name}")
        for column in self.df.columns:
            if column in ['class', 'path', self.gran_lvl_label_name]:
                print(column)
                new_df[column] = self.df[column]
        self.df = pd.DataFrame(new_df)
        print("Now clean df...")
        print(self.df)

        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        '''
        return:
            img: PIL object
            latent_lbl: scalar
            lbl: scalar
        '''
        img = Image.open(os.path.join(self.data_path, self.df.iloc[index]['path']))
        img = deepcopy(img)
        if not img.mode == 'RGB':
            img = img.convert("RGB")

        # labels
        if self.gran_lvl == '-1':
            lbl = -1
        else:
            
            lbl = int(self.df.iloc[index][self.gran_lvl_label_name])

        if self.transform:
            img = self.transform(img)

        return img, int(lbl)



class DynamicLabelDatasetIndex(DynamicLabelDataset):
    def __getitem__(self, index):
        '''
        return:
            img: PIL object
            latent_lbl: scalar
            lbl: scalar
        '''
        img = Image.open(os.path.join(self.data_path, self.df.iloc[index]['path']))
        img = deepcopy(img)
        if not img.mode == 'RGB':
            img = img.convert("RGB")

        # labels
        if self.gran_lvl == '-1':
            lbl = index
        else:
            # lbl = self.mapping[self.df.iloc[index][self.gran_lvl_label_name]] ## why do you need the mapping?
            lbl = int(self.df.iloc[index][self.gran_lvl_label_name])

        if self.transform:
            img = self.transform(img)

        return img, int(lbl), index

