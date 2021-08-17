import torch
import torch.nn.functional as F
from torch import nn

def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(inplace=True)
    )
    return blk

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

def get_nin(in_channels, out_channels):
    net = nn.Sequential(
        nin_block(in_channels, 96, kernel_size=5, stride=1, padding=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(96, 256, kernel_size=3, stride=1, padding=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(256, 384, kernel_size=3, stride=1, padding=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Dropout(0.5),
        nin_block(384, out_channels, kernel_size=3, stride=1, padding=1),
        GlobalAvgPool2d(),
        nn.Flatten()
    )
    return net

