import torch
import torch.nn as nn
from torch.nn import functional as F

class Conv2_1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1),
                 padding=(0, 0, 0), dilation=(1, 1, 1), bias=True):
        super().__init__()
        kernel_size = self.tuplify(kernel_size)
        stride = self.tuplify(stride)
        padding = self.tuplify(padding)
        dilation = self.tuplify(dilation)
        depth = kernel_size[0]
        slice_dim = (kernel_size[1] + kernel_size[2])//2
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Calculation from the paper to make number of parameters equal to 3D convolution
        self.hidden_size = int((depth * slice_dim**2 * in_channels * out_channels) / 
                               (slice_dim**2 * in_channels + depth * out_channels))

        self.conv2d = nn.Conv2d(in_channels, self.hidden_size, kernel_size[1:], stride[1:], padding[1:], bias=bias)
        self.conv1d = nn.Conv1d(self.hidden_size, out_channels, kernel_size[0], stride[0], padding[0], bias=bias)

    def forward(self, x):
        #2D convolution
        b, c, d, h, w = x.size()
        # Rearrange depth and channels and combine depth with batch size
        # This way each slice becomes an individual case to do 2D convolution on
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b*d, c, h, w)
        x = F.relu(self.conv2d(x))
        
        #1D convolution
        c, h, w = x.size()[1:]
        # Prepare for new rearrangement by parting depth and batch size
        x = x.view(b, d, c, h, w)
        # Rearrange shape to (batch size, height, width, channels, depth)
        # Combine batch size, height and width
        # This way each line of pixels depth-wise becomes an individual case
        x = x.permute(0, 3, 4, 2, 1).contiguous()
        x = x.view(b*h*w, c, d)
        x = self.conv1d(x)

        #Final output
        final_c, final_d = x.size()[1:]
        # Split batch, heigh and width again 
        x = x.view(b, h, w, final_c, final_d)
        # Rearrange dimensions back to the original order
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        return x
    
    @staticmethod
    def tuplify(arg):
        # Turns a single int i to a tuple (i, i, i)
        # If input is tuple/list it will assert length of 3 and return input
        if isinstance(arg, int):
            return (arg,) * 3
        assert isinstance(arg, tuple) or isinstance(arg, list), "Expected list or tuple, but got " + str(type(arg)) + "."
        assert len(arg) == 3, "Expected " + str(arg) + " to have 3 values!"
        return arg
