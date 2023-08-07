import torch.nn as nn

from mmcv.cnn import ConvModule
from ..registry import BACKBONES
#from ..builder import BACKBONES      #new version
from .resnet_ams import ResNetAMS

@BACKBONES.register_module()
class SnippetSampling_ResNetAMS(ResNetAMS):
    
    def __init__(self, 
                new_length=3,
                single_frame_channel=3,
                *args, 
                **kwargs):
        super().__init__(*args, **kwargs)
        self.new_length = new_length
        self.single_frame_channel = single_frame_channel
        self._reconstruct_first_layer()

    def _reconstruct_first_layer(self):
        print('Reconstructing first conv...')
        modules = list(self.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d),
                            list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (self.single_frame_channel*self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.repeat([1, self.new_length,] + [1]*(len(kernel_size[2:]))).contiguous()
        new_conv = nn.Conv2d(self.single_frame_channel*self.new_length, conv_layer.out_channels, 
                            conv_layer.kernel_size, conv_layer.stride, conv_layer.padding, 
                            bias=True if len(params)==2 else False)
        new_conv.weight.data = new_kernels

        if len(params) == 2:
            new_conv.bias.data = params[1].data
        layer_name = list(container.state_dict().keys())[0][:-7]

        setattr(container, layer_name, new_conv)


    def forward(self, x):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The feature of the input samples extracted
            by the backbone.
        """
        (BT, ch, h, w) = x.size()
        x = x.view(BT*ch, h, w)
        x = x.view((-1, self.single_frame_channel*self.new_length)+x.size()[-2:])

        x = self.conv1(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.temporal_block_indices:
                if i == 0:
                    #res2_CPTM
                    x = self.multiScale_res2(x)      
                elif i == 2:
                    #res4_CPTM
                    x = self.multiScale_res4(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)