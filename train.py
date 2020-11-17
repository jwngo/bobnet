import os 
import argparse
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler 
import numpy as np 
import torch.optim as optim
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt

from torch.utils import data 
from model import BobNet
from dataset import Adience

best_val_loss = 1e6
def parse_args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("exp_name", help="name of experiment to run")
    parser.add_argument("--efficientnet", action="store_true")
    args = parser.parse_args()
    return args
args = parse_args() 
class Trainer(object):
    def __init__(self, exp):
        self.path = os.path.join(os.getcwd(), 'gender_dataset')
        self.exp_name = exp
        self.device = torch.device('cuda:0') # Change to YAML 
        self.max_epochs = 50 
        self.batch_size = 50
        self.train_transform = transforms.Compose([
            transforms.Resize((227,227)),
            #transforms.RandomCrop((227,227)),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])
        self.train_dataset = Adience(
            path=self.path,
            image_set='train',
            transforms=self.train_transform,
        )
        self.train_loader = data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        self.val_dataset = Adience(
            path=self.path,
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
        self.iters_per_epoch = len(self.train_dataset) // self.batch_size 
        self.max_iters = self.max_epochs * self.iters_per_epoch
        # Use efficientnet
        if args.efficientnet:
            from efficientnet_pytorch import EfficientNet
            self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2, image_size=(227,227)).to(self.device)
        else:
            # BobNet
            self.model = BobNet(2).to(self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.001,
            momentum=0.9,
            weight_decay=1e-4,
        )
        self.criterion = nn.CrossEntropyLoss().cuda()
    def train(self, epoch, start_time): 
        iteration = epoch*self.iters_per_epoch if epoch > 0 else 0 
        epoch_loss = 0
        for batch_idx, sample in enumerate(self.train_loader): 
           iteration += 1
           img = sample['img'].to(self.device)
        #    plt.imshow(img[0].cpu().numpy().transpose(1,2,0))
        #    plt.show()
           gender = sample['gender'].to(self.device)
           outputs = self.model(img) 
           loss = self.criterion(outputs, gender) 

           self.optimizer.zero_grad() 
           loss.backward() 
           self.optimizer.step() 
           #self.lr_scheduler.step() 
           epoch_loss += loss.item()
           if iteration % 10 == 0: 
            print("Epoch: {:d}/{:d} || Iters: {:d}/{:d} || Loss: {:.4f}".format(epoch, self.max_epochs, iteration%self.iters_per_epoch, self.iters_per_epoch, loss.item()))
        if epoch%1 == 0: 
            save_dict = { 
                "epoch" : epoch, 
                "model" : self.model.state_dict(), 
                "optim" : self.optimizer.state_dict(), 
                }
            save_name = os.path.join(os.getcwd(), 'results', self.exp_name, 'run.pth'.format(epoch))
            torch.save(save_dict, save_name) 
            print("Model is saved: {}".format(save_name))
    def val(self, epoch): 
        global best_val_loss 
        epoch_loss = 0 
        with torch.no_grad(): 
            for batch_idx, sample in enumerate(self.val_loader): 
                img = sample['img'].to(self.device)
                label = sample['gender'].to(self.device) 
                outputs = self.model(img)
                loss = self.criterion(outputs, label)
                epoch_loss += loss.item() 
            print("Validation loss: {:.4f}".format(epoch_loss/len(self.val_loader)))

        # Save model if val_loss lower than best 
        if (epoch_loss/len(self.val_loader)) < best_val_loss:
            best_val_loss = epoch_loss/len(self.val_loader)
            save_name = os.path.join(os.getcwd(), 'results', self.exp_name, 'best_val_loss.pth') 
            save_dict = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "best_val_loss": best_val_loss, 
                }
            torch.save(save_dict, save_name) 
            print("val_loss is lower than best_val_loss! Model saved to {}".format(save_name))

if __name__ == '__main__':
    t = Trainer(args.exp_name)
    os.makedirs(os.path.join(os.getcwd(), 'results', t.exp_name), exist_ok=True)
    epoch = 0
    start_time = time.time()
    for epoch in range(t.max_epochs):
        t.train(epoch, start_time)
        t.val(epoch)

