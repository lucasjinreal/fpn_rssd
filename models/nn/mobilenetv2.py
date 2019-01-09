"""

Add MobileNetV2
construct layer components needed by SSD

extras:
regression_headers
classification_headers

Send them into SSD to construct the whole model
"""
import torch
from torch import nn
import math


# define some util blocks
def conv_bn(x, output, stride):
    return nn.Sequential(
        nn.Conv2d(x, output, 3, stride, 1, bias=False),
        nn.BatchNorm2d(output),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(x, output):
    return nn.Sequential(
        nn.Conv2d(x, output, 1, 1, 0, bias=False),
        nn.BatchNorm2d(output),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expand_ratio, dilation=1):
        """
        expand ratio should be 6
        dialation default to be 1
        stride should be 1 or 2
        """
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2], 'InsertedResidual stride must be 1 or 2, can not be changed'
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        # this convolution is the what we called Depth wise separable convolution
        # consist of pw and dw process, which is transfer channel and transfer shape in 2 steps
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, in_channels * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, 3, stride, padding=dilation, groups=in_channels*expand_ratio,
            dilation=dilation, bias=False),
            nn.BatchNorm2d(in_channels*expand_ratio),
            nn.ReLU6(inplace=True),
            # pw linear
            nn.Conv2d(in_channels*expand_ratio, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    implementation of MobileNetV2
    """

    def __init__(self, num_classes=20, input_size=224, width_mult=1.):
        """
        we just need classes, input_size and width_multiplier here

        the input_size must be dividable by 32, such as 224, 480, 512, 640, 960 etc.
        the width multiplier is width scale which can be 1 or less than 1
        we will judge this value to determine the last channel and input channel
        but why?

        Here is the reason for this:
        You can set input channel to 32, and the output of MobileNetV2 must be 1280
        so, when you multiply that channel, accordingly output should also be multiplied
        :param num_classes:
        :param input_size:
        :param width_mult:
        """
        super(MobileNetV2, self).__init__()

        assert input_size % 32 == 0, 'input_size must be divided by 32, such as 224, 480, 512 etc.'
        input_channel = int(32*width_mult)
        self.last_channel = int(1280*width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2)]

        # t:  c: channel, n: , s: stride
        self.inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        for t, c, n, s in self.inverted_residual_setting:
            output_channel = int(c*width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel

        # build last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))

        # this why input must can be divided by 32
        self.features.append(nn.AvgPool2d(int(input_size / 32)))
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, num_classes)
        )
        self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    n.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# for build MobileNetV2 dynamically
def get_inverted_residual_blocks(in_, out_, t=6, s=1, n=1):
    block = []
    block.append(InvertedResidual(in_, out_, expand_ratio=t, stride=s))
    for i in range(n-1):
        block.append(InvertedResidual(out_, out_, 1, t))
    return block


