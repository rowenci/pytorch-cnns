import torch
from torch import nn

def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU(inplace=True))
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)


def vgg(conv_arch, fc_features, fc_hidden_units):
    net = nn.Sequential()
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module("vgg_block_" + str(i + 1), vgg_block(num_convs, in_channels, out_channels))
    
    net.add_module("fc", nn.Sequential(
        nn.Flatten(),
        nn.Linear(fc_features, fc_hidden_units),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, fc_hidden_units),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, 10)
    ))
    return net
    
