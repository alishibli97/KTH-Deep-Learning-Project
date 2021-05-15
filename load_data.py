import os
from torch.utils.data import DataLoader
from segmentation_dataset import SegmentationDataset
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchvision import datasets, transforms

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f



train_path = "small_dataset/images/nir/"
train_labels_path = "small_dataset/labels/"

train_img_names_index = os.listdir(train_path)

labels_one_hot = {}
k = 9
for i, label in enumerate(listdir_nohidden(train_labels_path)):
    labels_one_hot[label] = np.zeros((k,))
    labels_one_hot[label][i] = 1

dataset = SegmentationDataset(train_img_names_index, labels_one_hot)
 
batch_size = 1
shuffle = True
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

image, label = next(iter(dataloader))
print(image)
print(label)
print(torch.unique(label))

plt.imshow(image[0], cmap='gray')
plt.show()
plt.imshow(label[0], cmap='gray')
plt.show()

print("finished")

