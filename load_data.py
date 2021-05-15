import os
from torch.utils.data import DataLoader
from segmentation_dataset import SegmentationDataset
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

train_path = "/Users/pontusolausson/supervised/Agriculture-Vision-2021/train/small/images/nir"
train_labels_path = "/Users/pontusolausson/supervised/Agriculture-Vision-2021/train/small/labels/"

train_img_index = os.listdir(train_path)

labels = {}
labels_one_hot = {}
for i, (root, dirs, filenames) in enumerate(os.walk(train_labels_path)):
    if (root != train_labels_path):
        label = root.replace(train_labels_path, "")
        labels[label] = filenames

        labels_one_hot[label] = [0 for j in range(9)]
        labels_one_hot[label][i - 1] = 1

dataset = SegmentationDataset(train_img_index, labels)

batch_size = 1
shuffle = True
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

image, label = next(iter(dataloader))
#for image_batch, label_batch in iter(dataloader):
#    print(image_batch)

print("finished")