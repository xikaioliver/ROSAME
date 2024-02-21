#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


class DigitConv(nn.Module):
    '''
    Convolutional neural network for MNIST digit recognition. From:
    https://github.com/pytorch/examples/blob/master/mnist/main.py
    '''
    def __init__(self, digit_class_num, input_channel=1):
        super(DigitConv, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, digit_class_num)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# In[40]:


class Encoder(nn.Module):
    def __init__(self, block_convnet, block_dim, block_size, hidden_dim, digit_class_num, prop_dim):
        super(Encoder, self).__init__()

        self.block_dim = block_dim
        self.hidden_dim = hidden_dim
        self.digit_class_num = digit_class_num
        self.prop_dim = prop_dim

        self.block_convnet = block_convnet

        self.mlp = nn.Sequential(
            nn.Linear(self.block_dim[0]*self.block_dim[1]*self.digit_class_num, self.hidden_dim*4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim*4, self.hidden_dim*2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.prop_dim)
        )

    def forward(self, x):
        pred = x.flatten(start_dim = 0, end_dim = 2)
        pred = self.block_convnet(pred)
        pred = pred.view(-1, self.block_dim[0]*self.block_dim[1]*self.digit_class_num)
        pred = self.mlp(pred)

        return F.sigmoid(pred)


# In[80]:


class Decoder(nn.Module):
    def __init__(self, block_dim, block_size, hidden_dim, prop_dim):
        super(Decoder, self).__init__()

        self.block_dim = block_dim
        self.block_size = block_size
        self.hidden_dim = hidden_dim
        self.prop_dim = prop_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim+self.prop_dim, self.hidden_dim*2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim*2, self.hidden_dim*2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim*2, self.hidden_dim*4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim*4, 16*self.block_dim[0]*self.block_dim[1]),
            nn.ReLU(),
        )

        self.block_convnet = nn.Sequential(
            nn.Linear(16, 500),
            nn.ReLU(),
            nn.Linear(500, 4*4*50),
            nn.ReLU(),
            nn.Unflatten(-1, (50, 4, 4)),
            nn.ConvTranspose2d(50, 50, 2, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(50, 20, 5, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(20, 20, 2, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(20, 1, 5, 1),
        )

    def forward(self, subsym, sym):
        pred = self.mlp(torch.cat((subsym, sym), 1))
        pred = pred.view(-1, 16)
        pred = self.block_convnet(pred)
        pred = pred.view(-1, self.block_dim[0], self.block_dim[1], 1, self.block_size, self.block_size)
        return pred


# In[ ]:




