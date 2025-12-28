import torch
import torch.nn as nn
from layers.conv_block import ConvBlock

class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels_list, out_channels_list, stride_list):
        super().__init__()
        self.num_scales = len(in_channels_list)
        self.blocks = nn.ModuleList()
        for s in range(self.num_scales):
            stride = stride_list[s]
            self.blocks.append(
                ConvBlock(in_channels_list[s], out_channels_list[s], stride=stride)
            )
    
    def forward(self, x_list):
        out_list = []
        for s, x in enumerate(x_list):
            out_list.append(self.blocks[s](x))
        return out_list
