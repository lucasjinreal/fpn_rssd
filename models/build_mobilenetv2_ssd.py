import torch.nn as nn

from .nn.mobiletnetv2 import MobileNetV2


# --------------------------- those method construct base components need for SSD
def create_mbv2_ssd(num_classes, is_test=False):
    base_net = MobileNetV2(num_classes).features
    # we should know the ouput of base_net

    # index of layer which we want extract out
    base_fea_idx = [12, 14]
    extras = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(1024, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.ReLU()
        )
    ])

    reg_head = nn.ModuleList([
        nn.Conv2d(512, 6*4, 3, 1),
        nn.Conv2d(512, 6*4, 3, 1),
        nn.Conv2d(512, 6*4, 3, 1),
        nn.Conv2d(512, 6*4, 3, 1),
        nn.Conv2d(512, 6*4, 3, 1),
    ])
    cls_head = nn.ModuleList([
        nn.Conv2d(512, 6*num_classes, 3, 1),
        nn.Conv2d(512, 6*num_classes, 3, 1),
        nn.Conv2d(512, 6*num_classes, 3, 1),
        nn.Conv2d(512, 6*num_classes, 3, 1),
        nn.Conv2d(512, 6*num_classes, 3, 1),
        nn.Conv2d(512, 6*num_classes, 3, 1),
        nn.Conv2d(512, 6*num_classes, 3, 1),
    ])