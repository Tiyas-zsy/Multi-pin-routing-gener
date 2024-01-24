import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import argparse
import os
import shutil
import numpy as np


def loss_fnn(k_sub,k_err,distance):
    mse_loss = nn.MSELoss()
    step = k_err * torch.sign(distance-1)+1
    loss_fn = mse_loss*(1+k_sub*step*distance)

class VAEgen(nn.Module):

    def __init__(self, hiddens:list =[32, 64], latent_dim=128) -> None:
        super().__init__()

        # encoder
        in_channels = 1
        modules = []
        img_length = 128
        #layer 1-6
        for cur_channels in hiddens:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,
                              cur_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1), 
                    
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2)
                    )
                    )
            '''nn.BatchNorm2d(cur_channels),#不知道加不加'''
            in_channels = cur_channels
            
        #layer7
        modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,
                              128,
                              kernel_size=3,
                              stride=1,
                              padding=1),
                    nn.ReLU()))
        
        self.img_length = 32
        #每个池化层会降低一半，128降低至32
        self.encoder = nn.Sequential(*modules)
        #高斯均值和方差
        self.latent_dim = latent_dim
        self.mean_linear = nn.Linear(128*32*32,self.latent_dim)
        self.var_linear = nn.Linear(128*32*32, self.latent_dim)
    def decoder(self,z):
        # decoder
        modules = []
        in_channels = 128
        decoder_projection = nn.Linear(self.latent_dim, 128*32*32)
        decoder_input_chw = (in_channels, self.img_length, self.img_length)
        d_hiddens = [64,32]
        in_channels = 128

        for cur_channals in  d_hiddens:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,
                              cur_channals,
                              kernel_size=3,
                              stride=1,
                              padding=1),
                    nn.Upsample(mode='bilinear', scale_factor=2)
                ))
            in_channels = cur_channals
 
        #final layer
        modules.append(
            nn.Sequential(
                #nn.Conv2d(in_channels,
                #           out_channels=32,
                #           kernel_size=3,
                #           stride=1,
                #           padding=0),
                #    nn.ReLU(),
                nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1),
                nn.Sigmoid()))
  
        decoder = nn.Sequential(*modules)
        x = decoder_projection(z)
        x = torch.reshape(x, (-1, *decoder_input_chw))
        decoded = decoder(x)
        return decoded

    def reparameterize(self, mu, logvar): # similar to sampling class in Keras code
        std = logvar.mul(0.5).exp_()

        eps = torch.normal(mu, std)

        z = mu + std * eps

        return z

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, 1)
        mean = self.mean_linear(encoded)
        logvar = self.var_linear(encoded)
        z = self.reparameterize(mean,logvar)
        '''x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))'''
        decoded = self.decoder(z)

        return decoded
        ''', mean, logvar'''

    def sample(self):
        z = torch.randn(1, self.latent_dim)
        '''x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        decoded = self.decoder(x)'''
        decoded = self.decoder(z)
        return decoded
    
    def loss_fnn(self,distance,y_pred,y_ture,k_sub= 0.001,k_err = 100):
        mse_loss = nn.MSELoss()
        loss = mse_loss(y_pred,y_ture)
        step = k_err * np.sign(distance-1)+1
        loss_fn = loss*(1+k_sub*step*distance)
        return loss_fn