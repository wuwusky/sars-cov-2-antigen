

import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional




def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=dilation, groups=groups, bias=False, dilation=dilation)
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

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
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

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
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(True)

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
        self.conv = nn.Conv1d(out_ch*7, out_ch, 1, 1, 0)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        d0 = self.d0(x)
        d1 = self.d1(d0)
        d2 = self.d2(d0)
        d3 = self.d3(d0)
        d4 = self.d4(d0)
        d5 = self.d5(d0)
        d6 = self.d6(d0)


        fp1 = d0 + d1
        fp2 = d0 + d1 + d2
        fp3 = d0 + d1 + d2 + d3
        fp4 = d0 + d1 + d2 + d3 + d4
        fp5 = d0 + d1 + d2 + d3 + d4 + d5
        fp6 = d0 + d1 + d2 + d3 + d4 + d5 + d6


        combine = torch.cat([d0, fp1, fp2, fp3, fp4, fp5, fp6], dim=1)

        # out = self.relu(self.bn(self.conv(combine)))
        out = self.conv(combine)

        return out

        
        




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

        self.conv1 = nn.Conv1d(4, self.inplanes, kernel_size=14, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # self.stem = nn.Sequential(
        #     nn.Conv1d(4, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.Conv1d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, dilation=2),
        #     nn.Conv1d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, dilation=4),
        #     nn.Conv1d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, dilation=8),
        #     nn.Conv1d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, dilation=16),
        # )
        self.stem = aspp_fpn_5(4, self.inplanes)

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.inplanes = 64
        self.dilation = 1
        # self.stem_ag = nn.Sequential(
        #     nn.Conv1d(1, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.Conv1d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, dilation=2),
        #     nn.Conv1d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, dilation=4),
        #     nn.Conv1d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, dilation=8),
        #     nn.Conv1d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, dilation=16),
        #     nn.Conv1d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, dilation=32),
        #     nn.Conv1d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, dilation=64),
        # )
        self.stem_ag = aspp_fpn_7(1, self.inplanes)


        self.layer1_ag = self._make_layer(block, 64, layers[0])
        self.layer2_ag = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3_ag = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4_ag = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])




        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(512 * block.expansion*2, num_classes)
        # self.sig = nn.Sigmoid()
        # self.drop = nn.Dropout(0.25)
        # self.embed = nn.Embedding(30, 32)

        self.head = nn.Sequential(
            # nn.Conv1d(512*block.expansion*2, 128, 1, 1, 0),
            # nn.Conv1d(128, num_classes, 1, 1, 0),
            nn.Flatten(start_dim=1),
            nn.Linear(512*block.expansion*2, 512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid(),
        )
        self.head_reg = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(512*block.expansion*2, 512),
            nn.Linear(512, 128),
            nn.Linear(128,1),
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
        x = self.stem(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.drop(x)
        x = self.layer2(x)
        # x = self.drop(x)
        x = self.layer3(x)
        # x = self.drop(x)
        x = self.layer4(x)
        # x = self.drop(x)

        x = self.avgpool(x)
        # x = torch.flatten(x, 1)


        return x
    
    def feat_extract_ag(self, x: Tensor):
        x = self.stem_ag(x)
        x = self.maxpool(x)

        x = self.layer1_ag(x)
        # x = self.drop(x)
        x = self.layer2_ag(x)
        # x = self.drop(x)
        x = self.layer3_ag(x)
        # x = self.drop(x)
        x = self.layer4_ag(x)
        # x = self.drop(x)

        x = self.avgpool(x)
        # x = torch.flatten(x, 1)


        return x
    

    def forward(self, x, x_ag) -> Tensor:
        # x_emb = self.embed(x.long())
        f = self.feat_extract(x)
        f_ag = self.feat_extract_ag(x_ag)

        # if self.training:
            # f = self.drop(f)
            # f_ag = self.drop(f_ag)

        f_all = torch.cat([f, f_ag], dim=1)

        # out = self.fc(f_all)
        # out = self.sig(out)
        out = self.head(f_all)
        return out
    
    def forward_reg(self, x, x_ag):
        f = self.feat_extract(x)
        f_ag = self.feat_extract_ag(x_ag)

        # if self.training:
            # f = self.drop(f)
            # f_ag = self.drop(f_ag)

        f_all = torch.cat([f, f_ag], dim=1)

        # out = self.fc(f_all)
        # out = self.sig(out)
        out = self.head_reg(f_all)

        return out



def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(**kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(BasicBlock, [2, 2, 2, 2],**kwargs)


def resnet34(**kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(BasicBlock, [3, 4, 6, 3],**kwargs)


def resnet50(**kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(Bottleneck, [3, 4, 6, 3],**kwargs)



def resnext50_32x4d(**kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(Bottleneck, [3, 4, 6, 3],**kwargs)


def resnext101_32x8d(**kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet(Bottleneck, [3, 4, 23, 3],**kwargs)


def wide_resnet50_2(**kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet(Bottleneck, [3, 4, 6, 3],**kwargs)


def wide_resnet101_2(**kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet(Bottleneck, [3, 4, 23, 3],**kwargs)





# # # temp_model = resnet50()
# # temp_input = torch.randn(2, 4, 64).long()
# temp_input_ag = torch.randn(2, 1, 1280)

# # temp_aspp = aspp_fpn_5(1, 8)

# # temp_out = temp_aspp(temp_input_ag)
# # print('test')

# # temp_embed = nn.Embedding(128, 128)
# temp_embed = nn.Linear(1280, 12)

# temp_out = temp_embed(temp_input_ag)

# # temp_out = temp_model(temp_input, temp_input_ag)
# print('test')