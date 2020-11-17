import os
import torch
import torch.nn as nn 
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 

from torch.utils import data 
from model import BobNet 
from dataset import Adience 

class Evaluator(object): 
    def __init__(self, exp): 
        self.path = os.path.join(os.getcwd(), 'gender_dataset')
        self.exp_name = exp
        self.batch_size = 128 
        self.device = torch.device('cuda:0') 
        self.val_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        self.val_dataset = Adience(
            path = self.path, 
            image_set='val',
            transforms=self.val_transform,
            )
        self.val_loader = data.DataLoader(
            dataset=self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=0, 
            pin_memory=True,
            drop_last=False, 
            )
        #self.model = BobNet(2).to(self.device) 
        from efficientnet_pytorch import EfficientNet
        self.model = EfficientNet.from_name('efficientnet-b0',num_classes=2, image_size=(227,227))
    def eval(self): 
        total = 0 
        correct = 0
        with torch.no_grad(): 
            for batch_idx, sample in enumerate(self.val_loader): 
                
                img = sample['img'].to(self.device) 
                print(img.shape)
                label = sample['gender'].to(self.device) 
                outputs = self.model(img)
                _, predicted = torch.max(outputs.data, 1) 
                total += label.size(0) 
                correct += (predicted == label).sum().item() 
        print('Accuracy of the network %f %%' %(100*correct/total))

if __name__ == '__main__':
    e = Evaluator('exp2')
    save_name = os.path.join(os.getcwd(), 'results', e.exp_name, 'best_val_loss.pth')
    save_dict = torch.load(save_name, map_location='cpu')
    print("Loading", save_name, "from Epoch {}:".format(save_dict['epoch']))
    e.model.load_state_dict(save_dict['model'])
    e.model = e.model.to(e.device)
    e.eval() 
