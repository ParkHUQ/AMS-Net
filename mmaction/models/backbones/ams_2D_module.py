import torch.nn as nn
import torch
from torch import Tensor
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional


class Temporal_Block(nn.Module):

    def __init__(
        self,
        channels: int,
    ) -> None:
        super(Temporal_Block, self).__init__()
        self.channels = channels

        #temporal
        self.conv = nn.Conv3d(self.channels, self.channels, kernel_size=(3, 1, 1), stride=1,
                                dilation=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn = nn.BatchNorm3d(self.channels)
        #self.bn = nn.SyncBatchNorm(self.channels)   #syncBN

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x : Tensor) -> Tensor:

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class MultiScale_Temporal_Block(nn.Module):

    def __init__(
        self,
        channels: int
    ) -> None:
        super(MultiScale_Temporal_Block, self).__init__()
        self.channels = channels

        self.Temporal_Block_1 = Temporal_Block(self.channels)
        self.Temporal_Block_2 = Temporal_Block(self.channels)
        self.Temporal_Block_3 = Temporal_Block(self.channels)

    def forward(self, x: Tensor) -> Tensor:

        #temporal
        output1 = self.Temporal_Block_1(x)
        output2 = self.Temporal_Block_2(output1)
        output3 = self.Temporal_Block_3(output2)
        output = (output1+output2+output3)/3

        return output

class MultiScale_Temporal(nn.Module):

    def __init__(
        self,
        channels: int,
        segments: int
    ) -> None:
        super(MultiScale_Temporal, self).__init__()
        self.channels = channels
        self.segments = segments

        self.temporal = MultiScale_Temporal_Block(self.channels)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view((-1, self.segments) + x.size()[1:])
        x = x.transpose(1, 2)

        #temporal
        output = self.temporal(x)

        output = output.transpose(1, 2)
        output = output.reshape((-1,) + output.size()[2:])

        return output


class Competitive_Progressive_Temporal_Module(nn.Module):
    def __init__(self, opt_block, inplance, branchs, rate, stride=1, L=32, gamma=1):
        """ Constructor
        Args:
            inplance: input channel dimensionality.
            branchs: the number of branchs.
            groups: num of convolution groups.
            rate: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(Competitive_Progressive_Temporal_Module, self).__init__()
        d = max(int(inplance / rate), L)
        self.branchs = branchs
        self.inplance = inplance
        self.gamma = gamma
        print("CompetitiveFusion", self.gamma)

        self.temporal_blocks = nn.ModuleList([])
        for i in range(self.branchs):
            self.temporal_blocks.append(
                opt_block(self.inplance)
            )

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(self.inplance, d)
        self.bn = nn.BatchNorm1d(d)
        #self.bn = nn.SyncBatchNorm(d)  # syncBN
        self.relu = nn.ReLU(inplace=True)

        self.fcs = nn.ModuleList([])
        for i in range(self.branchs):
            self.fcs.append(
                nn.Linear(d, self.inplance)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        fea = x
        for i, temporal_block in enumerate(self.temporal_blocks):
            fea = temporal_block(fea)
            pro_fea = fea.unsqueeze(dim=1)
            if i == 0:
                pro_feas = pro_fea
            else:
                pro_feas = torch.cat([pro_feas, pro_fea], dim=1)
        fea_U = torch.sum(pro_feas, dim=1)  
        fea_U = fea_U / self.branchs
        fea_s = self.avgpool(fea_U).view(fea_U.size(0), -1)   

        fea_h = self.fc(fea_s)
        fea_h = self.bn(fea_h)
        fea_h = self.relu(fea_h)

        for i, fc in enumerate(self.fcs):
            vector = fc(fea_h).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)

        #competitive fusion
        attention_vectors *= self.gamma
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  #b 3 c 1 1 1
        fea_v = (pro_feas * attention_vectors).sum(dim=1)
        fea_v = fea_v.transpose(1, 2)
        fea_v = fea_v.reshape((-1,) + fea_v.size()[2:])
        return fea_v


class CSTP_Stage1_Adaptive_Fusion(nn.Module):
    def __init__(self, inplance, branchs, rate, stride=1, L=32, gamma=1):
        """ Constructor
        Args:
            inplance: input channel dimensionality.
            branchs: the number of branchs.
            groups: num of convolution groups.
            rate: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(CSTP_Stage1_Adaptive_Fusion, self).__init__()
        d = max(int(inplance / rate), L)
        self.branchs = branchs
        self.inplance = inplance
        self.gamma = gamma
        print("CSTP_stage1", self.gamma)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(self.inplance, d)
        self.bn = nn.BatchNorm1d(d)
        #self.bn = nn.SyncBatchNorm(d)  # syncBN
        self.relu = nn.ReLU(inplace=True)

        self.fcs = nn.ModuleList([])
        for i in range(self.branchs):
            self.fcs.append(
                nn.Linear(d, self.inplance)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        flow_feas = []
        for fea in x:
            fea = fea.unsqueeze(dim=1)
            flow_feas.append(fea)
        flow_feas = torch.cat(flow_feas, dim=1)
        fea_U = torch.sum(flow_feas, dim=1) 
        fea_s = self.avgpool(fea_U).view(fea_U.size(0), -1)  
        fea_z = self.fc(fea_s)
        fea_z = self.bn(fea_z)
        fea_z = self.relu(fea_z)

        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1) 

        #collaborative fusion
        attention_vectors *= self.gamma
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  
        fea_v = flow_feas * attention_vectors      
        top_down_x, top_down_y, bottom_up_x, bottom_up_y = fea_v.split(split_size=1, dim=1)

        top_down_fea = torch.cat((top_down_x, top_down_y), dim=2).squeeze(1)
        bottom_up_fea = torch.cat((bottom_up_x, bottom_up_y), dim=2).squeeze(1)

        return top_down_fea, bottom_up_fea


class CSTP_Stage2_Adaptive_Fusion(nn.Module):
    def __init__(self, inplance, branchs, rate, stride=1, L=32, gamma=1):
        """ Constructor
        Args:
            inplance: input channel dimensionality.
            branchs: the number of branchs.
            groups: num of convolution groups.
            rate: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(CSTP_Stage2_Adaptive_Fusion, self).__init__()
        d = max(int(inplance / rate), L)
        self.branchs = branchs
        self.inplance = inplance
        self.gamma = gamma
        print("CSTP_stage2", self.gamma)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(self.inplance, d)
        self.bn = nn.BatchNorm1d(d)
        #self.bn = nn.SyncBatchNorm(d)  # syncBN
        self.relu = nn.ReLU(inplace=True)

        self.fcs = nn.ModuleList([])
        for i in range(self.branchs):
            self.fcs.append(
                nn.Linear(d, self.inplance)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        flow_feas = []
        for fea in x:
            fea = fea.unsqueeze(dim=1)
            flow_feas.append(fea)
        flow_feas = torch.cat(flow_feas, dim=1)
        fea_U = torch.sum(flow_feas, dim=1)  
        fea_s = self.avgpool(fea_U).view(fea_U.size(0), -1)   
        fea_z = self.fc(fea_s)
        fea_z = self.bn(fea_z)
        fea_z = self.relu(fea_z)

        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)

        #collaborative fusion
        attention_vectors *= self.gamma
        attention_vectors = self.softmax(attention_vectors)
        #[N,2,C,1,1,1]
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        #[N,2,1024,T,H,W]
        fea_v = flow_feas * attention_vectors
        top_down_fea, bottom_up_fea = fea_v.split(split_size=1, dim=1)
        #[N,2048,T,H,W] 
        fea_v = torch.cat((top_down_fea, bottom_up_fea), dim=2).squeeze(1)

        return fea_v 


class MultiScale_Temporal_Module(nn.Module):

    def __init__(
        self,
        channels: int,
        segments=8,
    ) -> None:
        super(MultiScale_Temporal_Module, self).__init__()
        self.channels = channels
        self.segments = segments

        self.cptm = Competitive_Progressive_Temporal_Module(Temporal_Block, self.channels, branchs=3, rate=16)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view((-1, self.segments) + x.size()[1:])
        x = x.transpose(1, 2)

        #temporal
        output = self.cptm(x)

        return output


class CPTM_Bottleneck(nn.Module):

    def __init__(
        self,
        channels: int,
        percent=0.25,
        gamma=1, 
        segments=8,
    ) -> None:
        super(CPTM_Bottleneck, self).__init__()
        self.channels = channels
        self.percent = percent
        self.current_channels = int(self.channels*self.percent)
        self.gamma = gamma
        self.segments = segments

        #reduce
        self.reduce = nn.Conv2d(self.channels, self.current_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_re = nn.BatchNorm2d(self.current_channels)
        #self.bn_re = nn.SyncBatchNorm(self.current_channels)  # syncBN

        #CPTM_BottleNeck
        self.cptm_bottleneck = Competitive_Progressive_Temporal_Module(Temporal_Block, self.current_channels, branchs=3, rate=16, gamma=self.gamma)

        #restore
        self.restore = nn.Conv2d(self.current_channels, self.channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_restore = nn.BatchNorm2d(self.channels)
        #self.bn_restore = nn.SyncBatchNorm(self.channels)  # syncBN
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.reduce(x)
        x = self.bn_re(x)
        x = self.relu(x)

        x = x.view((-1, self.segments) + x.size()[1:])
        x = x.transpose(1, 2)

        #CPTM
        output = self.cptm_bottleneck(x)

        output = self.restore(output)
        output = self.bn_restore(output)
        output = self.relu(output)

        return output

