import torch
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
import numpy as np
import os

class SegmentationDataset(Dataset):
    def __init__(self, img_names, one_hot, transform=None):
        self.img_names = img_names
        self.one_hot = one_hot
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        filename = self.img_names[index]
        input_ID = "small_dataset/images/nir/" + filename
        x = imread(input_ID)

        label_path = "small_dataset/labels/"
        y = None
        for label in self.one_hot:
            pre, ext = os.path.splitext(filename)
            yy = imread(label_path + label + "/" + pre + ".png")

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
            x, y = self.transform(x, y)

        x, y = torch.from_numpy(x).type(torch.float32), torch.from_numpy(y).type(torch.int64)

        return x, y
