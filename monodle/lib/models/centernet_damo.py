import os
import cv2
import torch
import torch.nn as nn
import numpy as np

from lib.backbones.backbones.tinynas_mob import TinyNAS
import ast
from lib.backbones.giraffe_fpn_btn import GiraffeNeckV2
        
class CenterNet3DDamo(nn.Module):
    def __init__(self, num_class=3, downsample=4):
        """
        CenterNet for monocular 3D object detection.
        :param backbone: the backbone of pipeline, such as dla34.
        :param neck: the necks of detection, such as dla_up.
        :param downsample: the ratio of down sample. [4, 8, 16, 32]
        :param head_conv: the channels of convolution in head. default: 256
        """
        assert downsample in [4, 8, 16, 32]
        super().__init__()

        self.heads = {'heatmap': num_class, 'offset_2d': 2, 'size_2d' :2, 'depth': 2, 'offset_3d': 2, 'size_3d':3, 'heading': 24}
        
        with open("/Users/sanchittanwar/Desktop/Workspace/courses/CS7643/final_project/methOD/monodle/lib/backbones/damoyolo_structure_small.txt", "r") as f:
            structure_info = f.read()   

        struct_str = ''.join([x.strip() for x in structure_info])
        struct_info = ast.literal_eval(struct_str)
        for layer in struct_info:
            if 'nbitsA' in layer:
                del layer['nbitsA']
            if 'nbitsW' in layer:
                del layer['nbitsW']
        
        self.backbone = TinyNAS(structure_info=struct_info, out_indices=(2,4,5),with_spp = True, depthwise=True)
        self.neck = GiraffeNeckV2(depth=0.5, hidden_ratio=0.5,  in_channels=[40, 80, 160], out_channels=[40, 80, 160], act='silu', spp=False, block_name='BasicBlock_3x3_Reverse', depthwise=True)

        channels = sum([40, 80, 160])
        for head in self.heads.keys():
            output_channels = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(channels, 256, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, output_channels, kernel_size=1, stride=1, padding=0, bias=True))

            # initialization
            if 'heatmap' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)


    def forward(self, input):
        feat = self.backbone(input)
        _, feat = self.neck(feat)
        ret = {}    
        for head in self.heads:
            ret[head] = self.__getattr__(head)(feat)
        return ret


    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)




if __name__ == '__main__':
    import torch
    
    net = CenterNet3DDamo(num_class=3)
    input = torch.randn(4, 3, 384, 1280)
    print(input.shape, input.dtype)
    output = net(input)
    print(output.keys())


