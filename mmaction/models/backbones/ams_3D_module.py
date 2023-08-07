import torch.nn as nn
import torch
from torch import Tensor
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from ams_2D_module import *


class CPTM_bottleneck(nn.Module):

    def __init__(
        self,
        channels: int,
        stage=1,
        percent=0.25,
        gamma=1,

        # default settings follow the ones in 2d net
        dim_restore=True,
        relu_out=True,
        short_cut = False,
        conv1d_relu_flag = True,

    ) -> None:
        super(CPTM_bottleneck, self).__init__()
        self.channels = channels
        self.percent = percent
        self.stage = stage//2
        self.current_channels = int(self.channels*self.percent/self.stage)
        self.gamma = gamma


        self.dim_restore = dim_restore
        self.relu_out = relu_out
        self.short_cut = short_cut
        self.conv1d_relu_flag = conv1d_relu_flag



        #reduce
        self.reduce = nn.Conv3d(self.channels, self.current_channels, kernel_size=1, stride=1, padding=0, bias=False)
        #self.bn_re = nn.BatchNorm3d(self.current_channels)
        self.bn_re = nn.SyncBatchNorm(self.current_channels)  # syncBN
        #CPTM
        self.cptm_bottleneck = \
            Competitive_Progressive_Temporal_Module(
                Temporal_Block, self.current_channels, branchs=3, rate=16, gamma=self.gamma,
                conv1d_relu_flag=self.conv1d_relu_flag)
        #restore
        if dim_restore:
            self.restore = nn.Conv3d(self.current_channels, self.channels, kernel_size=1, stride=1, padding=0, bias=False)
            #self.bn_restore = nn.BatchNorm3d(self.channels)
            self.bn_restore = nn.SyncBatchNorm(self.channels)  # syncBN
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:

        x = self.reduce(x)
        x = self.bn_re(x)
        x = self.relu(x)
        if self.short_cut :
            x_dim_reduce = x

        #CPTM
        output = self.cptm_bottleneck(x)

        if self.dim_restore:

            output = self.restore(output)
            output = self.bn_restore(output)

            if self.relu_out :
                output = self.relu(output)

        if self.short_cut :
            return output + x_dim_reduce
        else :
            return output

