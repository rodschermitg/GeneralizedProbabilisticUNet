import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from .utils import init_weights
import pdb

class DownConvBlock(nn.Module):
    """
    A block of three convolutional layers where each layer is followed by a non-linear activation function
    Between each block we add a pooling operation.
    """
    def __init__(self, input_dim, output_dim, initializers, padding, pool=True,norm=False, mc_dropout=False, dropout_rate=0.0):
        super(DownConvBlock, self).__init__()
        layers = []

        self.mc_dropout = mc_dropout

        if pool:
            layers.append(nn.AvgPool3d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

        layers.append(nn.Conv3d(input_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv3d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv3d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))

        if norm:
            layers.append(nn.BatchNorm3d(output_dim))

        if self.mc_dropout is True:
            self.dropout_op = nn.Dropout(p=dropout_rate)

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, patch):
        out = self.layers(patch)
        if self.mc_dropout is True:
            out = self.dropout_op(out)

        return out


class UpConvBlock(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If trilinear is set to false, we do a transposed convolution instead of upsampling
    """
    def __init__(self, input_dim, output_dim, initializers, padding, trilinear=True,norm=False, mc_dropout=False, dropout_rate=0.0):
        super(UpConvBlock, self).__init__()
        self.trilinear = trilinear
        if not self.trilinear:
            self.upconv_layer = nn.ConvTranspose3d(input_dim, output_dim, kernel_size=2, stride=2)
            self.upconv_layer.apply(init_weights)
#        pdb.set_trace()
        self.conv_block = DownConvBlock(input_dim, output_dim, initializers, padding, pool=False,norm=norm, mc_dropout=mc_dropout, dropout_rate=dropout_rate)

    def forward(self, x, bridge):
        if self.trilinear:
            up = nn.functional.interpolate(x, mode='trilinear', scale_factor=2, align_corners=True)
        else:
            up = self.upconv_layer(x)

        assert up.shape[3] == bridge.shape[3]
        out = torch.cat([up, bridge], 1)
        out =  self.conv_block(out)

        return out
