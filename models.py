## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # Output size = (64, 222, 222)
        self.conv1 = nn.Conv2d(1, 64, 3)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        # Output size = (64, 111, 111)
        self.maxPool = nn.MaxPool2d(2, 2)
        
        # Output size = (128, 109, 109) -> After maxPooling = (128, 54, 54)
        self.conv2 = nn.Conv2d(64, 128, 3)
        
        # Output size = (256, 52, 52) -> After maxPooling = (256, 26, 26)
        self.conv3 = nn.Conv2d(128, 256, 3)
        
        # Output size = (512, 24, 24) -> After maxPooling = (512, 12, 12)
        self.conv4 = nn.Conv2d(256, 512, 3)
        
        # Output size = (1024, 10, 10) -> After maxPooling = (1024, 5, 5)
        self.conv5 = nn.Conv2d(512, 1024, 3)
        
        # Output size = (2048, 3, 3) -> After maxPooling = (2048, 1, 1)
        self.conv6 = nn.Conv2d(1024, 2048, 3)
        
        self.fc1 = nn.Linear(2048, 1000)
        self.fc1_dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.maxPool(F.relu(self.conv1(x)))
        x = self.maxPool(F.relu(self.conv2(x)))
        x = self.maxPool(F.relu(self.conv3(x)))
        x = self.maxPool(F.relu(self.conv4(x)))
        x = self.maxPool(F.relu(self.conv5(x)))
        x = self.maxPool(F.relu(self.conv6(x)))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc1_dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
