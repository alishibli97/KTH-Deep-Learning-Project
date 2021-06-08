# dataloader code
from unet import *
from segmentation_dataset import SegmentationDataset
from torch.nn import DataParallel

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from loguru import logger

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm, trange
import random
import argparse
from configuration import config


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

class UnetTrainer:
    def __init__(self,cfg):
        self.cfg = cfg

        self.trainingAcc = []
        self.trainingLoss = []
        self.validationAcc = []
        self.validationLoss = []

        self.setup_network_params()
        self.setup_dataset()

    def setup_network_params(self):
        # SETTINGS
        Use_GPU = True
        self.Lr = self.cfg.lr # 1e-2
        self.channels = self.cfg.channels # 1 NIR vs 3 RGB
        self.classes = self.cfg.classes # 9
        self.maxEpochs = self.cfg.epochs # 30
        self.batch_size = self.cfg.batch # 64
        self.iter = self.cfg.iter # 3
        self.shuffle = True

        # Code 
        if Use_GPU: 
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                logger.info('cuda used')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

    def setup_dataset(self):
        """
        train_path =  "../Data/Agriculture-Vision-2021/train/images/nir/"
        val_path = "../Data/Agriculture-Vision-2021/val/images/nir/"
        test_path = "../Data/Agriculture-Vision-2021/test/images/nir/"

        train_labels_path = "../Data/Agriculture-Vision-2021/train/labels/"
        val_labels_path = "../Data/Agriculture-Vision-2021/val/labels/"
        test_labels_path = "../Data/Agriculture-Vision-2021/test/labels/"
        """

        """
        train_path = "small_dataset/images/nir/"
        val_path = "small_dataset/images/nir/"
        test_path = "small_dataset/images/nir/"

        train_labels_path = "small_dataset/labels/"
        val_labels_path = "small_dataset/labels/"
        test_labels_path = "small_dataset/labels/"
        """

        train_path = f"{self.cfg.data_dir}/images"
        val_path = f"{self.cfg.data_dir}/images"
        test_path = f"{self.cfg.data_dir}/images"

        train_labels_path = f"{self.cfg.data_dir}/labels/"
        val_labels_path = f"{self.cfg.data_dir}/labels/"
        test_labels_path = f"{self.cfg.data_dir}/labels/"

        train = os.listdir(train_path)
        val = os.listdir(val_path)
        test = os.listdir(test_path)

        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)

        train_img_names_index = train[:10000]
        val_img_names_index = val[:2000]
        test_img_names_index = test[:2000]

        labels_one_hot = {}
        k = self.cfg.classes # 8
        i=0
        for label in listdir_nohidden(train_labels_path):
            if label!="storm_damage":
                labels_one_hot[label] = np.zeros((k,))
                labels_one_hot[label][i] = 1
                i+=1

        train_dataset = SegmentationDataset("train", train_img_names_index, labels_one_hot, train_path, train_labels_path, use_cache=True)
        val_dataset = SegmentationDataset("validation", val_img_names_index, labels_one_hot, val_path, val_labels_path, use_cache=True)
        # test_dataset = SegmentationDataset("test", test_img_names_index, labels_one_hot, test_path, test_labels_path, use_cache=True)

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        # self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size)
    
    def initialize_model(self):
        # fix activationfunc, dropout and other settings for model as parameters later 
        self.model = UNet(self.channels, self.classes).to(self.device)
        self.model = DataParallel(self.model, device_ids=range(10),output_device=range(10))

        class_weights = torch.FloatTensor([1]+[5]*8).cuda()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.optimizer = torch.optim.SGD(self.model.parameters(), self.Lr)

    def itterProgress(x, text = "training"):
        return tqdm(enumerate(x), text, total = len(x))

    def run(self): 
        # itter = itterProgress(trainX)

        for epoch in range(self.maxEpochs):
            self.train(epoch)

            if epoch%self.iter==0:
                torch.save(self.model.state_dict(), f"trained_model_{epoch}.pth")

                f = open(f"history_{epoch}.txt","w")
                for i in range(len(self.trainingAcc)):
                    str = f"acc_train={self.trainingAcc[i]},acc_loss={self.trainingLoss[i]},val_acc={self.validationAcc[i]},val_loss={self.validationLoss[i]}\n"
                    f.write(str)

    def train(self):
        self.model.train()
        for i, (batch_x, batch_y) in enumerate(self.train_dataloader):
            indata, target = batch_x.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            indata = indata.unsqueeze(1)
            out = self.model(indata)
            out_softmax = torch.softmax(out, 1)
            img = self.postprocess(out_softmax)
            
            train_acc = self.iou(img, target)
            loss = self.criterion(out, target)
            train_loss = loss.item()
            
            self.trainingAcc.append(train_acc)
            self.trainingLoss.append(train_loss)

            loss.backward()
            self.optimizer.step()

            val_acc, val_loss = self.validate()

            self.validationAcc.append(val_acc)
            self.validationLoss.append(val_loss)

            logger.info(f"Epoch {self.epoch} batch {i+1}/{len(self.train_dataloader)} loss={train_loss} acc={train_acc} val_loss={val_loss} val_acc={val_acc}")

            if val_loss > np.mean(self.validationLoss)*1.5:
                logger.info("Overfitting detected")
                break

    def validate(self):
        self.model.eval()
        validationAcc_temp = []
        validationLoss_temp = []
        for i, (batch_x, batch_y) in enumerate(self.val_dataloader):
            indata, target = batch_x.to(self.device), batch_y.to(self.device)
            
            with torch.no_grad():
                indata = indata.unsqueeze(1)
                out = self.model.forward(indata)
                out_softmax = torch.softmax(out, 1)
                img = self.postprocess(out_softmax)
                
                val_acc = self.iou(img, target)            
                loss = self.criterion(out, target)
                val_loss = loss.item()

                validationAcc_temp.append(val_acc)
                validationLoss_temp.append(val_loss)
        
        return np.mean(validationAcc_temp),np.mean(validationLoss_temp)

    def postprocess(self,img):
        img = torch.argmax(img, dim=1)
        img = img.cpu().numpy()
        img = np.squeeze(img)
        img = torch.from_numpy(img).type(torch.int64)
        img = img.to(self.device)
        # img = re_normalize(img)
        return img

    def iou(self, prediction, target):
        eps = 0
        score = 0

        for k in range(1, 10):
            intersection = torch.sum((prediction==target) * (target==k)).item()
            union = torch.sum(prediction==k).item() + torch.sum(target==k).item()
            iou_k = 0 if intersection == 0 else (intersection + eps) / (union + eps)
            score += iou_k

        score = score / self.classes
        return score

def main():    
    cfg = config().parse_args()
    trainer = UnetTrainer()

    trainer.run()


if __name__=="__main__":
    main()