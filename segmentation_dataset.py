import torch
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
from skimage.transform import rescale
import cv2
import numpy as np
import os

import torchvision

class SegmentationDataset(Dataset):
    def __init__(self, img_names, one_hot, image_path, label_path, transform=None, use_cache=False, pre_transform=None):
        self.img_names = img_names
        self.one_hot = one_hot
        self.image_path = image_path
        self.label_path = label_path
        self.transform = transform
        self.use_cache = use_cache
        self.pre_transform = pre_transform

        if self.use_cache:
            self.cached_data = []

            for img_name in img_names:
                input_ID = self.image_path + img_name
                x = imread(input_ID)

                y = None
                try:
                    for label in self.one_hot:
                        pre, ext = os.path.splitext(img_name)
                        yy = imread(self.label_path + label + "/" + pre + ".png")

                        # Extract the label pixel value (+1 since 0 is the value for background pixels)
                        label_num = int(np.where(self.one_hot[label]==1)[0]) + 1

                        # Replace the white pixels (val=255) with the label pixel value
                        yy[yy==255] = label_num

                        # This is built on the assumption that no 2 labels can cover the same area of an image
                        if y is None:
                            y = yy
                        else:
                            y = yy
                    
                    if self.pre_transform is not None:
                        x, y = self.pre_transform(x, y)

                    x = cv2.resize(x, (128, 128), interpolation=cv2.INTER_NEAREST)
                    y = cv2.resize(y, (128, 128), interpolation=cv2.INTER_NEAREST)

                    self.cached_data.append((x, y))
                except:
                    pass

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):

        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            filename = self.img_names[index]
            input_ID = self.image_path + filename
            x = imread(input_ID)

            y = None
            for label in self.one_hot:
                pre, ext = os.path.splitext(filename)
                yy = imread(self.label_path + label + "/" + pre + ".png")

                # Extract the label pixel value (+1 since 0 is the value for background pixels)
                label_num = int(np.where(self.one_hot[label]==1)[0]) + 1

                # Replace the white pixels (val=255) with the label pixel value
                yy[yy==255] = label_num

                # This is built on the assumption that no 2 labels can cover the same area of an image
                if y is None:
                    y = yy
                else:
                    y += yy

        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)

        x, y = torch.from_numpy(x).type(torch.float32), torch.from_numpy(y).type(torch.int64)

        return x, y
