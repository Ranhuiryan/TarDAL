import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.models as models


class Generator(nn.Module):
    """
    Use to generate fused images.
    ir + vi -> fus
    """

    def __init__(self, dim: int = 64, depth: int = 3, fuse_scheme: int = 0):
        super(Generator, self).__init__()
        self.fuse_scheme = fuse_scheme # MIN, MAX, MEAN, SUM
        self.depth = depth
        resnet = models.resnet101(pretrained=True)
        for p in resnet.parameters():
            p.requires_grad = False
        self.resnet = resnet.conv1
        self.resnet.stride = 1
        self.resnet.padding = (0, 0)

        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * (i + 1), dim, (3, 3), (1, 1), 1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU()
            ) for i in range(depth)
        ])

        self.fuse = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(dim * (depth + 1), dim * 4, (3, 3), (1, 1), 1, bias=False),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(dim * 4, dim * 2, (3, 3), (1, 1), 1, bias=False),
                nn.BatchNorm2d(dim * 2),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(dim * 2, dim, (3, 3), (1, 1), 1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(dim, 1, (3, 3), (1, 1), 1, bias=False),
                nn.Tanh()
            ),
        )

        # Initialize parameters for other parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def tensor_min(self, tensors):
        min_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                min_tensor = tensor
            else:
                min_tensor = torch.min(min_tensor, tensor)
        return min_tensor
    
    def tensor_max(self, tensors):
        max_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                max_tensor = tensor
            else:
                max_tensor = torch.max(max_tensor, tensor)
        return max_tensor

    def tensor_sum(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        return sum_tensor

    def tensor_mean(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        mean_tensor = sum_tensor / len(tensors)
        return mean_tensor

    def operate(self, operator, tensors):
        out_tensors = []
        for tensor in tensors:
            out_tensor = operator(tensor)
            out_tensors.append(out_tensor)
        return out_tensors

    def tensor_padding(self, tensors, padding=(1, 1, 1, 1), mode='constant', value=0):
        out_tensors = []
        for tensor in tensors:
            # if the channel is 1, then add two more dimension
            if tensor.shape[1] == 1:
                tensor = torch.cat([tensor, tensor, tensor], dim=1)
            out_tensor = F.pad(tensor, padding, mode=mode, value=value)
            out_tensors.append(out_tensor)
        return out_tensors

    def forward(self, ir: Tensor, vi: Tensor) -> Tensor:
        outs = self.tensor_padding(tensors=[ir, vi], padding=(3, 3, 3, 3), mode='replicate')
        outs = self.operate(self.resnet, outs)
        for i in range(self.depth):
            t = self.operate(self.encoder[i], outs)
            for j in range(len(outs)):
                outs[j] = torch.cat([outs[j], t[j]], dim=1)
        
        if self.fuse_scheme == 0:
            fus = self.tensor_min(outs)
        elif self.fuse_scheme == 1:
            fus = self.tensor_max(outs)
        elif self.fuse_scheme == 2:
            fus = self.tensor_sum(outs)
        elif self.fuse_scheme == 3:
            fus = self.tensor_mean(outs)
        else:
            fus = self.tensor_sum(outs)
        
        fus = self.fuse(fus)

        return fus
