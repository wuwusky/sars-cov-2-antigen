import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn.functional as F
import random

# temp_1 = torch.randn(2, 128, 1)
# temp_2 = torch.bmm(temp_1, temp_1.permute(0,2,1))
# pass

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=dilation, groups=groups, bias=False, dilation=dilation)
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv3x3_2D(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=dilation, groups=groups, bias=False, dilation=dilation)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)



def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv1x1_2D(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock_2D(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock_2D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3_2D(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3_2D(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes
        if inplanes != planes:
            self.identity_conv = nn.Conv2d(inplanes, planes, 1, 1, 0)


    def forward(self, x: Tensor) -> Tensor:
        if self.inplanes == self.planes:
            identity = x
        else:
            identity = self.identity_conv(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        # self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        # self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        # self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class dia_conv(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, stride=1, d=1):
        super().__init__()
        padding = int((ksize-1)/2)*d
        self.conv = nn.Conv1d(in_ch, out_ch, ksize, stride=stride, dilation=d, padding=padding)
        # self.conv_1 = nn.Conv1d(out_ch, out_ch, 1, 1, 0)
    def forward(self, input):
        out = self.conv(input)
        # out = self.conv_1(out)
        return out




class aspp_fpn_5(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(aspp_fpn_5, self).__init__()

        self.d0 = dia_conv(in_ch, out_ch, 3, 1, 1)
        self.d1 = dia_conv(out_ch, out_ch, 3, 1, 2)
        self.d2 = dia_conv(out_ch, out_ch, 5, 1, 4)
        self.d3 = dia_conv(out_ch, out_ch, 9, 1, 8)
        self.d4 = dia_conv(out_ch, out_ch, 17, 1, 16)
        self.conv = nn.Conv1d(out_ch*5, out_ch, 1, 1, 0)
        # self.bn = nn.BatchNorm1d(out_ch)
        # self.relu = nn.PReLU(True)

    def forward(self, x):
        d0 = self.d0(x)
        d1 = self.d1(d0)
        d2 = self.d2(d0)
        d3 = self.d3(d0)
        d4 = self.d4(d0)


        fp1 = d0 + d1
        fp2 = d0 + d1 + d2
        fp3 = d0 + d1 + d2 + d3
        fp4 = d0 + d1 + d2 + d3 + d4


        combine = torch.cat([d0, fp1, fp2, fp3, fp4], dim=1)

        # out = self.relu(self.bn(self.conv(combine)))
        out = self.conv(combine)

        return out


# class aspp_fpn_7(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(aspp_fpn_7, self).__init__()

#         self.d0 = dia_conv(in_ch, out_ch, 3, 1, 1)
#         self.d1 = dia_conv(out_ch, out_ch, 3, 1, 2)
#         self.d2 = dia_conv(out_ch, out_ch, 5, 1, 4)
#         self.d3 = dia_conv(out_ch, out_ch, 9, 1, 8)
#         self.d4 = dia_conv(out_ch, out_ch, 17, 1, 16)
#         self.d5 = dia_conv(out_ch, out_ch, 33, 1, 32)
#         self.d6 = dia_conv(out_ch, out_ch, 65, 1, 64)

#         self.combine = nn.Sequential(
#             nn.Conv1d(out_ch*7, out_ch*7, 1, 1, 0),
#             nn.BatchNorm1d(out_ch*7),
#             nn.ReLU(True),
#             nn.Conv1d(out_ch*7, out_ch, 1, 1, 0),
#         )


#     def forward(self, x):
#         d0 = self.d0(x)
#         d1 = self.d1(d0)
#         d2 = self.d2(d0)
#         d3 = self.d3(d0)
#         d4 = self.d4(d0)
#         d5 = self.d5(d0)
#         d6 = self.d6(d0)


#         fp1 = d0 + d1
#         fp2 = d0 + d1 + d2
#         fp3 = d0 + d1 + d2 + d3
#         fp4 = d0 + d1 + d2 + d3 + d4
#         fp5 = d0 + d1 + d2 + d3 + d4 + d5
#         fp6 = d0 + d1 + d2 + d3 + d4 + d5 + d6


#         combine = torch.cat([d0, fp1, fp2, fp3, fp4, fp5, fp6], dim=1)

#         # out = self.relu(self.bn(self.conv(combine)))
#         out = self.combine(combine)

#         return out


class aspp_fpn_7(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(aspp_fpn_7, self).__init__()

        self.d0 = dia_conv(in_ch, out_ch, 3, 1, 1)
        self.d1 = dia_conv(out_ch, out_ch, 3, 1, 2)
        self.d2 = dia_conv(out_ch, out_ch, 5, 1, 4)
        self.d3 = dia_conv(out_ch, out_ch, 9, 1, 8)
        self.d4 = dia_conv(out_ch, out_ch, 17, 1, 16)
        self.d5 = dia_conv(out_ch, out_ch, 33, 1, 32)
        self.d6 = dia_conv(out_ch, out_ch, 65, 1, 64)

        self.combine = nn.Sequential(
            # nn.Conv1d(out_ch*7, out_ch*7, 3, 1, 1, groups=7),
            # nn.BatchNorm1d(out_ch*7),
            # nn.ReLU(True),
            nn.Conv1d(out_ch*7, out_ch, 3, 1, 1),
        )

        self.dropout = nn.Dropout(p=0.25, inplace=True)


    def forward(self, x):
        d0 = self.d0(x)
        # d0 = self.dropout(d0)
        d1 = self.d1(d0)
        # d1 = self.dropout(d0)
        d2 = self.d2(d0)
        # d2 = self.dropout(d0)
        d3 = self.d3(d0)
        # d3 = self.dropout(d0)
        d4 = self.d4(d0)
        # d4 = self.dropout(d0)
        d5 = self.d5(d0)
        # d5 = self.dropout(d0)
        d6 = self.d6(d0)
        # d6 = self.dropout(d0)


        fp1 = d0 + d1
        fp2 = d0 + d1 + d2
        fp3 = d0 + d1 + d2 + d3
        fp4 = d0 + d1 + d2 + d3 + d4
        fp5 = d0 + d1 + d2 + d3 + d4 + d5
        fp6 = d0 + d1 + d2 + d3 + d4 + d5 + d6


        combine = torch.cat([d0, fp1, fp2, fp3, fp4, fp5, fp6], dim=1)

        # out = self.relu(self.bn(self.conv(combine)))
        out = self.combine(combine)

        return out


class aspp_fpn_3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(aspp_fpn_3, self).__init__()

        self.d0 = dia_conv(in_ch, out_ch, 3, 1, 1)
        self.d1 = dia_conv(out_ch, out_ch, 3, 1, 2)
        self.d2 = dia_conv(out_ch, out_ch, 5, 1, 4)
        
        self.conv = nn.Conv1d(out_ch*3, out_ch, 1, 1, 0)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.SELU(True)

    def forward(self, x):
        d0 = self.d0(x)
        d1 = self.d1(d0)
        d2 = self.d2(d0)



        fp1 = d0 + d1
        fp2 = d0 + d1 + d2



        combine = torch.cat([d0, fp1, fp2], dim=1)

        out = self.conv(combine)

        return out       



class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = max_pool_layer

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)



class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 5,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)


        self.stem = aspp_fpn_7(4, self.inplanes)
        self.stem_non = NONLocalBlock1D(self.inplanes, 16)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])


        self.inplanes = 64
        self.dilation = 1
        self.stem_ag = aspp_fpn_7(1, self.inplanes)
        self.stem_non_ag = NONLocalBlock1D(self.inplanes, 16)
        self.layer1_ag = self._make_layer(block, 64, layers[0])
        self.layer2_ag = self._make_layer(block, 128, layers[1])
        self.layer3_ag = self._make_layer(block, 256, layers[2])
        self.layer4_ag = self._make_layer(block, 512, layers[3])


        self.stem_sp = aspp_fpn_7(4, 64)

        self.stem_ag_sp = aspp_fpn_7(1, 64)

        self.avgpool = nn.AdaptiveAvgPool1d(1)


        self.head = nn.Sequential(
            # nn.Conv1d(512*block.expansion*2, 1024, 1, 1, 0),
            # # nn.BatchNorm1d(1024),
            # nn.Dropout(0.25),
            # nn.PReLU(),
            # nn.Conv1d(1024, 512, 1, 1, 0),
            # # nn.BatchNorm1d(512),
            # nn.Dropout(0.25),
            # nn.PReLU(),
            # nn.Conv1d(512, 128, 1, 1, 0),
            # # nn.BatchNorm1d(128),
            # nn.Dropout(0.25),
            # nn.PReLU(),
            # nn.Conv1d(128, num_classes, 1, 1, 0),
            # nn.Flatten(start_dim=1),
            # nn.Sigmoid(),


            nn.Flatten(start_dim=1),
            nn.Linear(512 * block.expansion*2, 1024),
            # nn.Dropout(0.95),
            nn.PReLU(),
            nn.Linear(1024, 512),
            # nn.Dropout(0.95),
            nn.PReLU(),
            nn.Linear(512, 128),
            # nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid(),
        )
            




        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def feat_extract(self, x: Tensor):
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x
    
    def feat_extract_ag(self, x: Tensor):
        x = self.pool(x)
        x = self.layer1_ag(x)
        x = self.layer2_ag(x)
        x = self.layer3_ag(x)
        x = self.layer4_ag(x)
        x = self.avgpool(x)
        return x
    

    def forward(self, x, x_ag) -> Tensor:
        # x_emb = self.embed(x.long())
        # [329:583]
        x_emb = self.stem(x[:,:4,:])
        x_sp_emb = self.stem_sp(x[:,4:,:])
        x_emb = self.stem_non(x_emb+x_sp_emb)


        x_ag_emb = self.stem_ag(x_ag[:,:1,329:583])
        x_ag_sp_emb = self.stem_ag_sp(x_ag[:,1:,329:583])
        x_ag_emb = self.stem_non_ag(x_ag_emb+x_ag_sp_emb)

        f = self.feat_extract(x_emb)
        f_ag = self.feat_extract_ag(x_ag_emb)
        # f_ag = self.feat_extract(x_ag_emb)

        f_all = torch.cat([f, f_ag], dim=1)
        out = self.head(f_all)
        return out


class ResNet_trans(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 5,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet_trans, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)


        self.stem = aspp_fpn_7(4, self.inplanes)
        
        temp_layer = nn.TransformerEncoderLayer(64, 8, activation=F.gelu, batch_first=True)
        self.seq_trans = nn.TransformerEncoder(temp_layer, 4)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])


        self.inplanes = 64
        self.dilation = 1
        self.stem_ag = aspp_fpn_7(1, self.inplanes)
        
        temp_layer = nn.TransformerEncoderLayer(64, 8, activation=F.gelu, batch_first=True)
        self.seq_trans_ag = nn.TransformerEncoder(temp_layer, 4)
        self.layer1_ag = self._make_layer(block, 64, layers[0])
        self.layer2_ag = self._make_layer(block, 128, layers[1])
        self.layer3_ag = self._make_layer(block, 256, layers[2])
        self.layer4_ag = self._make_layer(block, 512, layers[3])


        self.stem_sp = aspp_fpn_7(4, 64)

        self.stem_ag_sp = aspp_fpn_7(1, 64)

        self.avgpool = nn.AdaptiveAvgPool1d(1)


        self.head = nn.Sequential(
            # nn.Conv1d(512*block.expansion*2, 1024, 1, 1, 0),
            # # nn.BatchNorm1d(1024),
            # nn.Dropout(0.25),
            # nn.PReLU(),
            # nn.Conv1d(1024, 512, 1, 1, 0),
            # # nn.BatchNorm1d(512),
            # nn.Dropout(0.25),
            # nn.PReLU(),
            # nn.Conv1d(512, 128, 1, 1, 0),
            # # nn.BatchNorm1d(128),
            # nn.Dropout(0.25),
            # nn.PReLU(),
            # nn.Conv1d(128, num_classes, 1, 1, 0),
            # nn.Flatten(start_dim=1),
            # nn.Sigmoid(),


            nn.Flatten(start_dim=1),
            nn.Linear(512 * block.expansion*2, 1024),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.95),
            nn.PReLU(),
            nn.Linear(512, 128),
            # nn.Dropout(0.25),
            nn.PReLU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid(),
        )
            




        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def feat_extract(self, x: Tensor):
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x
    
    def feat_extract_ag(self, x: Tensor):
        x = self.pool(x)
        x = self.layer1_ag(x)
        x = self.layer2_ag(x)
        x = self.layer3_ag(x)
        x = self.layer4_ag(x)
        x = self.avgpool(x)
        return x
    

    def forward(self, x, x_ag) -> Tensor:
        # x_emb = self.embed(x.long())
        # [329:583]
        x_emb = self.stem(x[:,:4,:])
        x_sp_emb = self.stem_sp(x[:,4:,:])
        x_emb = torch.permute((x_emb + x_sp_emb), (0,2,1))
        x_emb = self.seq_trans(x_emb)
        x_emb = x_emb.permute(0,2,1)

        x_ag_emb = self.stem_ag(x_ag[:,:1,329:583])
        x_ag_sp_emb = self.stem_ag_sp(x_ag[:,1:,329:583])
        x_ag_emb = torch.permute((x_ag_emb + x_ag_sp_emb), (0,2,1))
        x_ag_emb = self.seq_trans_ag(x_ag_emb)
        x_ag_emb = x_ag_emb.permute(0,2,1)

        f = self.feat_extract(x_emb)
        f_ag = self.feat_extract_ag(x_ag_emb)
        # f_ag = self.feat_extract(x_ag_emb)

        f_all = torch.cat([f, f_ag], dim=1)
        out = self.head(f_all)
        return out


class conv_wh(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_wh, self).__init__()
        self.conv_w = nn.Conv2d(in_ch, out_ch, (3, 1), 1, (1,0))
        self.conv_h = nn.Conv2d(in_ch, out_ch, (1, 3), 1, (0,1))
        self.conv_comb = nn.Conv2d(out_ch*2, out_ch, 3, 1, 1)
        # self.bn = nn.BatchNorm2d(out_ch)
        # self.act = nn.ReLU(True)
    def forward(self, input):
        # out_w = self.act(self.bn(self.conv_w(input))) 
        # out_h = self.act(self.bn(self.conv_h(input))) 
        # out_wh = self.act(self.bn(self.conv_wh(input)))

        # out = self.conv_comb(torch.cat([out_w, out_h, out_wh], dim=1))
        out_w = self.conv_w(input)
        out_h = self.conv_h(input)
        # out_wh = self.conv_wh(input)

        out = self.conv_comb(torch.cat([out_w, out_h], dim=1))

        return out

class coor_head_sim(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.25):
        super(coor_head_sim, self).__init__()
        self.conv = conv_wh(in_ch, in_ch//4)
        self.conv_wh = nn.Sequential(
            nn.Conv2d(in_ch//4, in_ch//4, (3,3), 1, (1,1)),
            nn.BatchNorm2d(in_ch//4),
            nn.ReLU(True),
            nn.Conv2d(in_ch//4, in_ch//2, (3,3), 1, (1,1)),
            nn.BatchNorm2d(in_ch//2),
            nn.ReLU(True),
            nn.Conv2d(in_ch//2, out_ch, (3,3), 1, (1,1)),
        )
        self.act = nn.Sigmoid()
        self.drop = dropout
        if dropout > 0:
            self.dropout = nn.Dropout2d(dropout)

    
    def forward(self, x):
        if self.drop > 0:
            x = self.dropout(x)
        f = self.conv(x)
        # print(f.shape)
        f_wh = self.conv_wh(f)
        out = self.act(f_wh)
        return out




class ResNet_mt(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 5,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet_mt, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.stem_VH = aspp_fpn_7(1, self.inplanes)
        self.stem_VH_sp = aspp_fpn_7(1, self.inplanes)
        self.layer1_VH = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2_VH = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_VH = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_VH = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5_VH = self._make_layer(block, 512, layers[3], stride=2)

        # self.inplanes = 64
        # self.dilation = 1
        # self.stem_VL = aspp_fpn_7(1, self.inplanes)
        # self.stem_VL_sp = aspp_fpn_7(1, self.inplanes)
        # self.layer1_VL = self._make_layer(block, 64, layers[0], stride=2)
        # self.layer2_VL = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3_VL = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4_VL = self._make_layer(block, 512, layers[3], stride=2)
        
        self.inplanes = 64
        self.dilation = 1
        self.stem_ag = aspp_fpn_7(1, self.inplanes)
        self.stem_ag_sp = aspp_fpn_7(1, self.inplanes)
        self.layer1_ag = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2_ag = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_ag = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_ag = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5_ag = self._make_layer(block, 512, layers[3], stride=2)
        self.layer6_ag = self._make_layer(block, 512, layers[3], stride=2)
        self.layer7_ag = self._make_layer(block, 512, layers[3], stride=2)


        temp_layer = nn.TransformerEncoderLayer(64, 8, activation='gelu', batch_first=True, dropout=0.0)
        self.seq_trans = nn.TransformerEncoder(temp_layer, 4)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        feature_dim = 512
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(feature_dim*block.expansion*3, 512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid(),
        )

        self.head_VH2CH = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(feature_dim*block.expansion, 512),
            # nn.Dropout(0.5),
            nn.Linear(512, 2),
            nn.Sigmoid(),
            # nn.SELU()
        )

        self.head_VL2CL = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(feature_dim*block.expansion, 512),
            # nn.Dropout(0.5),
            nn.Linear(512, 2),
            nn.Sigmoid(),
            # nn.SELU()
        )

        self.head_map = coor_head_sim(64, 1, 0.25)
            

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def feat_extract_vh(self, x: Tensor):
        x = self.layer1_VH(x)
        x = self.layer2_VH(x)
        x = self.layer3_VH(x)
        x = self.layer4_VH(x)
        x = self.layer5_VH(x)
        x = self.avgpool(x)
        return x
    
    def feat_extract_vl(self, x: Tensor):
        x = self.layer1_VH(x)
        x = self.layer2_VH(x)
        x = self.layer3_VH(x)
        x = self.layer4_VH(x)
        x = self.layer5_VH(x)
        x = self.avgpool(x)
        return x

    def feat_extract_ag(self, x: Tensor):
        x = self.layer1_ag(x)
        x = self.layer2_ag(x)
        x = self.layer3_ag(x)
        x = self.layer4_ag(x)
        x = self.layer5_ag(x)
        x = self.layer6_ag(x)
        x = self.layer7_ag(x)
        x = self.avgpool(x)
        return x
    
    def forward(self, x, x_ag):
        # x_vh = x[:,0:1,:]
        # x_vl = x[:,1:2,:]
        # x_ch = x[:,2:3,:64]
        # x_cl = x[:,3:4,:64]

        x_emb_vh = self.stem_VH(x[:, 0:1, :])
        x_emb_vl = self.stem_VH(x[:, 1:2, :])
        x_emb_vh_sp = self.stem_VH_sp(x[:,5:6,:])
        x_emb_vl_sp = self.stem_VH_sp(x[:,6:7,:])

        x_emb_vh = x_emb_vh + x_emb_vh_sp
        x_emb_vl = x_emb_vl + x_emb_vl_sp

        x_emb_vh = torch.permute(x_emb_vh, (0,2,1))
        x_emb_vh = self.seq_trans(x_emb_vh)
        x_emb_vh = x_emb_vh.permute(0,2,1)
        x_emb_vl = torch.permute(x_emb_vl, (0,2,1))
        x_emb_vl = self.seq_trans(x_emb_vl)
        x_emb_vl = x_emb_vl.permute(0,2,1)


        # x_emb_ch = self.stem_CH(x[:, 2:3, :])
        # x_emb_cl = self.stem_CL(x[:, 3:4, :])

        # x_ag_emb = self.stem_ag(x_ag[:,:1,329:583])
        # x_ag_emb_sp = self.stem_ag_sp(x_ag[:,1:,329:583])
        x_ag_emb = self.stem_ag(x_ag[:,:1,:])
        x_ag_emb_sp = self.stem_ag_sp(x_ag[:,1:,:])
        x_ag_emb = x_ag_emb + x_ag_emb_sp

        x_ag_emb = torch.permute(x_ag_emb, (0,2,1))
        x_ag_emb = self.seq_trans(x_ag_emb)
        x_ag_emb = x_ag_emb.permute(0,2,1)

        
        f_vh = self.feat_extract_vh(x_emb_vh)
        f_vl = self.feat_extract_vl(x_emb_vl)
        f_ag = self.feat_extract_ag(x_ag_emb)

        f_all = torch.cat([f_vh, f_vl, f_ag], dim=1)
        out = self.head(f_all)
        return out
    
    def forward_train_mt(self, x, x_ag):
        # x_vh = x[:,0:1,:]
        # x_vl = x[:,1:2,:]
        # x_ch = x[:,2:3,:64]
        # x_cl = x[:,3:4,:64]

        x_emb_vh = self.stem_VH(x[:, 0:1, :])
        x_emb_vl = self.stem_VH(x[:, 1:2, :])
        x_emb_vh_sp = self.stem_VH_sp(x[:,5:6,:])
        x_emb_vl_sp = self.stem_VH_sp(x[:,6:7,:])

        x_emb_vh = x_emb_vh + x_emb_vh_sp
        x_emb_vl = x_emb_vl + x_emb_vl_sp

        x_emb_vh = torch.permute(x_emb_vh, (0,2,1))
        x_emb_vh = self.seq_trans(x_emb_vh)
        x_emb_vh = x_emb_vh.permute(0,2,1)
        x_emb_vl = torch.permute(x_emb_vl, (0,2,1))
        x_emb_vl = self.seq_trans(x_emb_vl)
        x_emb_vl = x_emb_vl.permute(0,2,1)


        # x_emb_ch = self.stem_CH(x[:, 2:3, :])
        # x_emb_cl = self.stem_CL(x[:, 3:4, :])

        # x_ag_emb = self.stem_ag(x_ag[:,:1,329:583])
        # x_ag_emb_sp = self.stem_ag_sp(x_ag[:,1:,329:583])
        x_ag_emb = self.stem_ag(x_ag[:,:1,:])
        x_ag_emb_sp = self.stem_ag_sp(x_ag[:,1:,:])
        x_ag_emb = x_ag_emb + x_ag_emb_sp

        x_ag_emb = torch.permute(x_ag_emb, (0,2,1))
        x_ag_emb = self.seq_trans(x_ag_emb)
        x_ag_emb = x_ag_emb.permute(0,2,1)

        f_vh = self.feat_extract_vh(x_emb_vh)
        f_vl = self.feat_extract_vl(x_emb_vl)
        f_ag = self.feat_extract_ag(x_ag_emb)

        ## aux task
        loc_cdrh = self.head_VH2CH(f_vh)
        loc_cdrl = self.head_VL2CL(f_vl)

        ## final task
        feature_list = [f_vh, f_vl, f_ag]
        # random.shuffle(feature_list)
        f_all = torch.cat(feature_list, dim=1)
        out = self.head(f_all)

        return out, loc_cdrh, loc_cdrl
    
    def forward_train_str(self, x_str):
        x_emb_vh = self.stem_VH(x_str)
        # x_emb_vl = self.stem_VH(x_str)
        x_ag_emb = self.stem_ag(x_str)

        x_emb_vh = torch.permute(x_emb_vh, (0,2,1))
        x_emb_vh = self.seq_trans(x_emb_vh)
        # x_emb_vh = x_emb_vh.permute(0,2,1)
        # x_emb_vl = torch.permute(x_emb_vl, (0,2,1))
        # x_emb_vl = self.seq_trans(x_emb_vl)
        # x_emb_vl = x_emb_vl.permute(0,2,1)
        x_ag_emb = torch.permute(x_ag_emb, (0,2,1))
        x_ag_emb = self.seq_trans(x_ag_emb)
        # x_ag_emb = x_ag_emb.permute(0,2,1)

        x_emb_vh = torch.unsqueeze(x_emb_vh, dim=3)
        f_h_map = torch.matmul(x_emb_vh.permute(0,2,1,3), x_emb_vh.permute(0,2,3,1))
        # x_emb_vl = torch.unsqueeze(x_emb_vl, dim=3)
        # f_l_map = torch.matmul(x_emb_vl.permute(0,2,1,3), x_emb_vl.permute(0,2,3,1))
        x_ag_emb = torch.unsqueeze(x_ag_emb, dim=3)
        f_a_map = torch.matmul(x_ag_emb.permute(0,2,1,3), x_ag_emb.permute(0,2,3,1))

        out_h = self.head_map(f_h_map)
        # out_l = self.head_map(f_l_map)
        out_a = self.head_map(f_a_map)

        return out_h, out_a




def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model

def _resnet_mt(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any
):
    model = ResNet_mt(block, layers, **kwargs)
    return model

def _resnet_trans(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any
):
    model = ResNet_trans(block, layers, **kwargs)
    return model

def resnet18(**kwargs: Any) -> ResNet:
    return _resnet(BasicBlock, [2, 2, 2, 2],**kwargs)
def resnet50(**kwargs: Any) -> ResNet:
    return _resnet(Bottleneck, [3, 4, 6, 3],**kwargs)

####################################
def resnet18_trans(**kwargs: Any) -> ResNet:
    return _resnet_trans(BasicBlock, [2, 2, 2, 2],**kwargs)

def resnet50_trans(**kwargs: Any):
    return _resnet_trans(Bottleneck, [3, 4, 6, 3],**kwargs)


###################################
def resnet18_mt(**kwargs: Any) -> ResNet:
    return _resnet_mt(BasicBlock, [2, 2, 2, 2],**kwargs)

def resnet50_mt(**kwargs: Any):
    return _resnet_mt(Bottleneck, [3, 4, 6, 3],**kwargs)



# def resnet101(**kwargs: Any) -> ResNet:
#     return _resnet(Bottleneck, [3, 4, 23, 3],**kwargs)

# temp_model = resnet50_trans()
# temp_input = torch.randn(2, 8, 128)
# temp_input_ag = torch.randn(2, 2, 1280)

# # # temp_aspp = aspp_fpn_5(1, 8)

# # # temp_out = temp_aspp(temp_input_ag)
# # # print('test')

# # # temp_embed = nn.Embedding(128, 128)
# # temp_embed = nn.Linear(1280, 12)

# # temp_out = temp_embed(temp_input_ag)

# temp_out = temp_model(temp_input, temp_input_ag)
# print('test')