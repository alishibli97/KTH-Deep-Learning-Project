import torch
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread

class SegmentationDataset(Dataset):
    def __init__(self, inputs, targets, transform=None):
        self.inputs = inputs
        self.targets = targets['waterway']
        self.transform = transform
        # self.inputs_dtype = torch.float32
        # self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_ID = "/Users/pontusolausson/supervised/Agriculture-Vision-2021/train/small/images/nir/" + self.inputs[index]
        target_ID = "/Users/pontusolausson/supervised/Agriculture-Vision-2021/train/small/labels/waterway/" + self.targets[index]

        x, y = imread(input_ID), imread(target_ID)

        if self.transform is not None:
            x, y = self.transform(x, y)

        # x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y
