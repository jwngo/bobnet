import os
import numpy as np
import torch
import cv2

from torch.utils import data
from PIL import Image

class Adience(data.Dataset):
    def __init__(self, path, image_set, transforms=None):
        super(Adience, self).__init__()
        assert image_set in ('train', 'val'), "image_set is not valid!"
        self.data_path = path
        self.image_set = image_set
        self.transforms = transforms
        # self.data_path = os.path.join(os.getcwd(), 'gender_dataset')
        self.createIndex()

    def createIndex(self):
        self.img_list = []
        self.img_path = []
        self.label_list = []
        if self.image_set == 'val':
            listfile = os.path.join(self.data_path, 'fold_4_data.txt')
            with open(listfile) as f:
                next(f) # Skip header
                for line in f:
                    line = line.split('\t')
                    userid = line[0]
                    faceid = line[2] # Not sure what faceid is used for 
                    imgid = faceid + '.' + line[1]
                    # age = line[3] # We don't need age yet 
                    gender = line[4]
                    if gender == '' or gender == 'u':
                        continue # Remove images without specific gender
                    filepath = os.path.join(os.getcwd(), 'gender_dataset', 'aligned',  userid)
                    self.img_path.append(filepath)
                    self.img_list.append(imgid)
                    self.label_list.append(gender)
        if self.image_set == 'train':
            for i in range(4):
                listfile = os.path.join(self.data_path, 'fold_{}_data.txt'.format(i))
                with open(listfile) as f:
                    next(f) # Skip header 
                    for line in f:
                        line = line.split('\t')
                        userid = line[0]
                        faceid = line[2]
                        imgid = faceid + '.' + line[1]
                        gender = line[4]
                        if gender == '' or gender == 'u':
                            continue # Remove images without specific gender
                        filepath = os.path.join(os.getcwd(), 'gender_dataset', 'aligned',  userid)
                        self.img_path.append(filepath)
                        self.img_list.append(imgid)
                        self.label_list.append(gender)

    def __getitem__(self, idx):
        path = self.img_path[idx]
        filename = ''
        items = os.listdir(path)
        for item in items:
            if item.endswith(self.img_list[idx]):
                filename+=item
                break
        img = cv2.imread(os.path.join(path,filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = Image.fromarray(img)
        gender = self.label_list[idx]
        assert gender in ('m', 'f'), "No gender for this image!"
        if gender == 'm':
            gender = 0
        elif gender == 'f':
            gender = 1

        if self.transforms is not None:
            img = self.transforms(img)
        sample = {
            'img': img,
            'gender': gender,
        }
        return sample 
    
    def __len__(self):
        return len(self.img_list)
