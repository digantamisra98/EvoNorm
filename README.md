# Evolving Normalization-Activation Layers

*Google AI and DeepMind*

- [x] Implement EvoNorm S0 and B0 with Training Mode support
- [x] Solve Shape Error with group_std and instance_std functions
- [x] Solve NaN Error Issue with S0
- [ ] Fix Error with shape in running variance calculation

<div style="text-align:center"><img src ="figures/evonorm.PNG"  width="1000"/></div>
<p>
<em>Figure 1. Left: Computation graph of a searched normalization activation layer that is batch-independent, named EvoNorm-S0.      Right: ResNet-50 results with EvoNorm-S0 as the batch size over 8 workers varies from 1024 to 32 on ImageNet. EvoNorm-S0 also outperforms both BN and GN-based layers on MobileNetV2 and Mask R-CNN.
</em>
</p>

## Usage: 

### S0:

Basic Block of ResNet:

```
from evonorm2d import EvoNorm2D
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = FeatureMix(inplanes, planes, stride = stride, name = 'conv3x3')
        self.evo = EvoNorm2D(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = FeatureMix(planes, planes, groups =1, name = 'conv3x3')
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.evo(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
```
