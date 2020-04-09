import torch
import torch.nn as nn

def instance_std(x, eps=1e-5):
    var = torch.var(x, dim = (2, 3), keepdim=True)
    return torch.sqrt(var + eps)

def group_std(x, groups = 32, eps = 1e-5):
    N, C, H, W = x.size()
    x = torch.reshape(x, (N, groups, C // groups, H, W))
    var = torch.var(x, dim = (2, 3, 4), keepdim = True).expand_as(x)
    return torch.reshape(torch.sqrt(var + eps), (N, C, H, W))

class EvoNorm2D(nn.Module):

    def __init__(self, input, non_linear = True, version = 'S0', momentum = 0.9, training = True):
        super(EvoNorm2D, self).__init__()
        self.non_linear = non_linear
        self.version = version
        self.training = training
        self.momentum = momentum
        if self.version not in ['B0', 'S0']:
            raise ValueError("Invalid EvoNorm version")
        self.insize = input
        self.gamma = nn.Parameter(torch.ones(1, self.insize, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, self.insize, 1, 1))
        if self.non_linear:
            self.v = nn.Parameter(torch.ones(1,self.insize,1,1))
        self.register_buffer('running_var', torch.ones(self.insize))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_var.fill_(1)
    
    def forward(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(x.dim()))
        if self.version == 'S0':
            if self.non_linear:
                num = x * torch.sigmoid(self.v * x)
                return num / group_std(x) * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta
        if self.version == 'B0':
            exponential_average_factor = self.momentum
            if self.training:
                var = x.var([0, 2, 3], unbiased=False)
                n = x.numel() / x.size(1)
                with torch.no_grad():
                    self.running_var = exponential_average_factor * var * n / (n - 1)\
                        + (1 - exponential_average_factor) * self.running_var
            else:
                var = self.running_var
            if self.non_linear:
                den = torch.max(var, self.v * x + instance_std(x))
                return x / den * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta
