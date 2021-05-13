import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        mid_channels = 64

        # common
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(p=0.5,inplace=True)

        # downsampling
        self.conv1 = nn.Conv2d(n_channels*1, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_channels*2, mid_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(n_channels*4, mid_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(n_channels*8, mid_channels, kernel_size=3, padding=1)
        
        # mid
        self.conv_mid = nn.Conv2d(n_channels*16, mid_channels, kernel_size=3, padding=1)

        # upsampling
        self.deconv4 = nn.ConvTranspose2d(n_channels*8 , mid_channels, kernel_size=3, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(n_channels*4 , mid_channels, kernel_size=3, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(n_channels*2 , mid_channels, kernel_size=3, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(n_channels*1 , mid_channels, kernel_size=3, stride=2, padding=1)

        # output layer
        self.conv_out = nn.Conv2d(1, mid_channels, kernel_size=1, padding=1)

    def forward(self, x):
        # downsampling
        conv1 = self.conv1(x)
        conv1 = self.relu(conv1)
        conv1 = self.conv1(conv1)
        conv1 = self.relu(conv1)
        pool1 = self.pool(conv1)
        pool1 = self.drop(pool1)

        conv2 = self.conv2(pool1)
        conv2 = self.relu(conv2)
        conv2 = self.conv2(conv2)
        conv2 = self.relu(conv2)
        pool2 = self.pool(conv2)
        pool2 = self.drop(pool2)

        conv3 = self.conv3(pool2)
        conv3 = self.relu(conv3)
        conv3 = self.conv3(conv3)
        conv3 = self.relu(conv3)
        pool3 = self.pool(conv3)
        pool3 = self.drop(pool3)

        conv4 = self.conv4(pool3)
        conv4 = self.relu(conv4)
        conv4 = self.conv4(conv4)
        conv4 = self.relu(conv4)
        pool4 = self.pool(conv4)
        pool4 = self.drop(pool4)

        # mid
        conv_mid = self.conv_mid(pool4)
        conv_mid = self.relu(conv_mid)
        conv_mid = self.conv_mid(conv_mid)
        conv_mid = self.relu(conv_mid)

        # upsampling
        deconv4 = self.deconv4(conv_mid)
        uconv4 = torch.cat([deconv4, conv4], dim=1)
        uconv4 = self.drop(uconv4)
        uconv4 = self.conv4(uconv4)
        uconv4 = self.relu(uconv4)
        uconv4 = self.conv4(uconv4)
        uconv4 = self.relu(uconv4)

        deconv3 = self.deconv3(uconv4)
        uconv3 = torch.cat([deconv3, conv3], dim=1)
        uconv3 = self.drop(uconv3)
        uconv3 = self.conv3(uconv3)
        uconv3 = self.relu(uconv3)
        uconv3 = self.conv3(uconv3)
        uconv3 = self.relu(uconv3)

        deconv2 = self.deconv2(uconv3)
        uconv2 = torch.cat([deconv2, conv2], dim=1)
        uconv2 = self.drop(uconv2)
        uconv2 = self.conv2(uconv2)
        uconv2 = self.relu(uconv2)
        uconv2 = self.conv2(uconv2)
        uconv2 = self.relu(uconv2)

        deconv1 = self.deconv1(uconv2)
        uconv1 = torch.cat([deconv1, conv1], dim=1)
        uconv1 = self.drop(uconv1)
        uconv1 = self.conv1(uconv1)
        uconv1 = self.relu(uconv1)
        uconv1 = self.conv1(uconv1)
        uconv1 = self.relu(uconv1)

        # output layer
        out = self.conv_out(uconv1)

        return out