import os
from torch.utils.data import DataLoader
from segmentation_dataset import SegmentationDataset
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchvision import datasets, transforms

train_path = "small_dataset/images/nir/"
train_labels_path = "small_dataset/labels/"

train_img_index = os.listdir(train_path)

labels = {}
labels_one_hot = {}
k = 0
for i, (root, dirs, filenames) in enumerate(os.walk(train_labels_path)):
    if i == 0:
        k = len(dirs)

    if (root != train_labels_path):
        label = root.replace(train_labels_path, "")
        labels[label] = filenames

        labels_one_hot[label] = np.zeros((k,))
        labels_one_hot[label][i - 1] = 1

dataset = SegmentationDataset(train_img_index, labels, labels_one_hot)

batch_size = 1
shuffle = True
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

#image, label = next(iter(dataloader))
#print(label)

for image_batch, label_batch in iter(dataloader):
    print(torch.max(label_batch))

print("finished")