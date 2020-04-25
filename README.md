# Evolving Normalization-Activation Layers

*Google AI and DeepMind*

- [x] Implement EvoNorm S0 and B0 with Training Mode support
- [x] Solve Shape Error with group_std and instance_std functions
- [x] Solve NaN Error Issue with S0
- [x] Fix Error with shape in running variance calculation in EvoNorm B0
- [x] Solve NaN Error Issue with B0

<div style="text-align:center"><img src ="figures/evonorm.PNG"  width="1000"/></div>
<p>
<em>Figure 1. Left: Computation graph of a searched normalization activation layer that is batch-independent, named EvoNorm-S0.      Right: ResNet-50 results with EvoNorm-S0 as the batch size over 8 workers varies from 1024 to 32 on ImageNet. EvoNorm-S0 also outperforms both BN and GN-based layers on MobileNetV2 and Mask R-CNN.
</em>
</p>

## Usage: 

```
from evonorm2d import EvoNorm2D
# For B0 version
evoB0 = EvoNorm2D(input, affine = True, version = 'B0', training = True)

# For S0 version 
evoS0 = EvoNorm2D(input)
```

