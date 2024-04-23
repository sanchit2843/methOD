import os
import cv2
import torch
import torch.nn as nn
import numpy as np

from lib.backbones import dla
from lib.backbones.dlaup import DLAUp
from lib.backbones.hourglass import get_large_hourglass_net
from torchvision.ops import RoIAlign

# from lib.backbones.hourglass import load_pretrian_model

## head which generates 3d from 2d proposals:
## head will flatten the roi pooled feature and first apply 2 fc layer to get to a 512 dim feature vector
## This 512 feature will be passed to 4 different fc for prediction, these prediction includes: box dim regression, rotation regression and logits, localization regression


class ROIBox3DHead(nn.Module):
    def __init__(self, in_size=128 * 7 * 7, hidden_size=512, num_bins=24):
        super().__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dim_predictor = nn.Linear(hidden_size, 3)
        nn.init.normal_(self.dim_predictor.weight, std=0.001)
        for l in [
            self.dim_predictor,
        ]:
            nn.init.constant_(l.bias, 0)

        self.rot_cls_predictor = nn.Linear(hidden_size, num_bins)
        self.rot_reg_predictor = nn.Linear(hidden_size, num_bins)
        nn.init.normal_(self.rot_cls_predictor.weight, std=0.001)
        nn.init.normal_(self.rot_reg_predictor.weight, std=0.001)
        for l in [
            self.rot_cls_predictor,
            self.rot_reg_predictor,
        ]:
            nn.init.constant_(l.bias, 0)

        self.loc_predictor = nn.Linear(hidden_size, 3)
        nn.init.normal_(self.loc_predictor.weight, std=0.001)
        for l in [
            self.loc_predictor,
        ]:
            nn.init.constant_(l.bias, 0)

    def forward(self, features):
        features = features.view(features.size(0), -1)
        features = self.relu(self.fc1(features))
        features = self.relu(self.fc2(features))
        dim = self.dim_predictor(features)
        rot_cls = self.rot_cls_predictor(features)
        rot_reg = self.rot_reg_predictor(features)
        loc = self.loc_predictor(features)
        return dim, rot_cls, rot_reg, loc


class CenterNet3DProposal(nn.Module):
    def __init__(self, backbone="dla34", neck="DLAUp", num_class=3, downsample=4):
        """
        CenterNet for monocular 3D object detection.
        :param backbone: the backbone of pipeline, such as dla34.
        :param neck: the necks of detection, such as dla_up.
        :param downsample: the ratio of down sample. [4, 8, 16, 32]
        :param head_conv: the channels of convolution in head. default: 256
        """
        assert downsample in [4, 8, 16, 32]
        super().__init__()

        self.heads = {
            "heatmap": num_class,
            "offset_2d": 2,
            "size_2d": 2,
            "depth": 2,
            "offset_3d": 2,
            "size_3d": 3,
            "heading": 24,
        }
        self.backbone_rgb = getattr(dla, backbone)(pretrained=True, return_levels=True)
        self.backbone_hha = getattr(dla, backbone)(pretrained=True, return_levels=True)

        channels = (
            self.backbone_rgb.channels
        )  # channels list for feature maps generated by backbone
        self.first_level = int(np.log2(downsample))
        scales = [2**i for i in range(len(channels[self.first_level :]))]
        self.neck_rgb = DLAUp(channels[self.first_level :], scales_list=scales)
        self.neck_hha = DLAUp(channels[self.first_level :], scales_list=scales)
        # feature fusion [such as DLAup, FPN]
        self.roi_align = RoIAlign(
            output_size=(7, 7), spatial_scale=1.0 / downsample, sampling_ratio=1
        )

        for head in self.heads.keys():
            output_channels = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(
                    2 * channels[self.first_level],
                    256,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    256, output_channels, kernel_size=1, stride=1, padding=0, bias=True
                ),
            )
            if "heatmap" in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)
        ## make copy of heads with different weights without using __setattr__
        self.head_roi_align = ROIBox3DHead()

    def forward(self, rgb, hha, proposals=None):
        """_summary_

        Args:
            rgb (_type_): _description_
            hha (_type_): _description_
            proposals (_type_, optional): Make sure proposals are of shape K*5 where first index is the batch index. Defaults to None.

        Returns:
            _type_: _description_
        """
        feat_rgb = self.backbone_rgb(rgb)
        feat_rgb = self.neck_rgb(feat_rgb[self.first_level :])
        feat_hha = self.backbone_hha(hha)
        feat_hha = self.neck_hha(feat_hha[self.first_level :])
        feat = torch.cat([feat_rgb, feat_hha], dim=1)

        ## TODO: Implement RoIAlign layer
        roi_align_feats = self.roi_align(feat, proposals)
        ret = {}
        for head in self.heads:
            a = self.__getattr__(head)(feat)
            ret[head] = a

        if proposals is None:
            return ret

        (dim, rot_cls, rot_reg, loc) = self.head_roi_align(roi_align_feats)
        ### return losses if training

        return ret, (dim, rot_cls, rot_reg, loc)

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    import torch

    net = CenterNet3DProposal(backbone="dla34")
    input = torch.randn(4, 3, 384, 1280)
    proposals = torch.randn(4, 4)
    proposals = [proposals] * 4
    ## convert this to K*5 where first index is the batch index
    for i in range(len(proposals)):
        proposals[i] = torch.cat(
            [torch.ones(proposals[i].shape[0], 1) * i, proposals[i]], dim=1
        )
    proposals = torch.cat(proposals, dim=0)
    print(proposals.shape)
    output = net(input, input, proposals)
    print(output.keys())
