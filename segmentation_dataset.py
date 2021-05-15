import torch
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, inputs, targets, one_hot, transform=None):
        self.inputs = inputs
        self.targets = targets
        self.one_hot = one_hot
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_ID = "small_dataset/images/nir/" + self.inputs[index]
        x = imread(input_ID)

        label_path = "small_dataset/labels/"
        y = None
        for label in self.targets:
            yy = imread(label_path + label + "/" + self.targets[label][index])

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

        # x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y
