import json
import os

import imageio
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES = ['01Gorilla', '02Unicorn', '03Mallard', '04Turtle', '05Whale', '06Bird', '07Owl', '08Sabertooth', '09Swan',
               '10Sheep', '11Pig', '12Zalika', '13Pheonix', '14Elephant', '15Parrot', '16Cat', '17Scorpion', '18Obesobeso', '19Bear', '20Puppy']

class LEGODataset(Dataset):
    """
    test set    
    """
    def __init__(self, dataset_path, class_name='15', resize=256, isTrainingSet=False, device="cpu"):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(
            class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.size = resize
        self.isTrainingSet = isTrainingSet
        self.tfms = transforms.Compose([
            transforms.Resize(self.size), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform_mask = transforms.Compose([
                        transforms.Resize(self.size, Image.NEAREST),
                        transforms.ToTensor()
        ])

        if isTrainingSet:
            self.x = self.load_dataset_folder()
        else:
            self.x, self.y, self.mask = self.load_dataset_folder()

    def __getitem__(self, idx):
        """
        x: <Tensor> shape==[1, 1, self.size, self.size]
        y: <Tensor> shape==[1]  ground truth
        mask: <Tensor> shape==[1, 1, self.size, self.size]
        xpath: <str>
        maskpath: <str>
        """
        x = self.x[idx]
        
        xpath = x
        x = Image.open(x).convert("RGB")
        x = self.tfms(x)
        if self.isTrainingSet:
            return x.to("cpu"), xpath
        
        y, mask = self.y[idx], self.mask[idx]
        # x = x.resize((self.size, self.size), Image.NEAREST)
        if y == 0:  # good
            mask = torch.zeros([self.size, self.size])
            mask = mask.unsqueeze(0)
            mask_path = ""
        else:
            # import pdb; pdb.set_trace()
            
            mask_path = mask
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
            # mask = imageio.imread(mask, cv2.IMREAD_GRAYSCALE)     #原这里imread写成read
            # # mask = cv2.resize(x, (self.size, self.size), interpolation=cv2.INTER_AREA)    #x?????
            # mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_AREA)   #应该是这个resize产生了非0和255的值

        return x.to("cpu"), xpath, y, mask.to("cpu"), mask_path

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        if self.isTrainingSet:
            img_dir = os.path.join(self.dataset_path, self.class_name, 'train', 'good')
            img_fpath_list = sorted([os.path.join(img_dir, f)
                                    for f in os.listdir(img_dir)
                                    if f.endswith('.png')])
            x = []
            x.extend(img_fpath_list)
            return list(x)
        else:
            phase = 'test'
            x, y, mask = [], [], []

            img_dir = os.path.join(self.dataset_path, self.class_name, phase)
            gt_dir = os.path.join(
                self.dataset_path, self.class_name, 'ground_truth')
            img_types = sorted(os.listdir(img_dir))
            for img_type in img_types:

                img_type_dir = os.path.join(img_dir, img_type)
                if not os.path.isdir(img_type_dir):
                    continue
                img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                        for f in os.listdir(img_type_dir)
                                        if f.endswith('.png')])
                x.extend(img_fpath_list) # test path

                if img_type == 'good':
                    y.extend([0] * len(img_fpath_list))
                    mask.extend([None] * len(img_fpath_list))
                else:
                    y.extend([1] * len(img_fpath_list))
                    gt_type_dir = os.path.join(gt_dir, img_type)
                    img_fname_list = [os.path.splitext(os.path.basename(f))[
                        0] for f in img_fpath_list]
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                    for img_fname in img_fname_list]
                    mask.extend(gt_fpath_list)

            assert len(x) == len(y), 'number of x and y should be same'

            return list(x), list(y), list(mask)
