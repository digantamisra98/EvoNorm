import torch
import torch.nn as nn

def instance_std(x, eps=1e-5):
    var = torch.std(x, dim = (2, 3), keepdim=True)
    return torch.sqrt(var + eps)

def group_std(x, groups = 32, eps = 1e-5):
    N, C, H, W = x.size()
    x = torch.reshape(x, (N, groups, C // groups, H, W))
    var = torch.std(x, dim = (3, 4, 2), keepdim = True)
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

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
    
    def forward(self, x):
        if self.version == 'S0':
            if self.non_linear:
                num = x * torch.sigmoid(self.v * x)
                return num / group_std(x) * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta
        if self.version == 'B0':
            if self.training:
                var = x.var([0,2,3])
                self.running_var = (self.momentum * self.running_var) + (1.0-self.momentum) * (x.shape[0]/(x.shape[0]-1)*var)
            else:
                var = self.running_var
            sigma = var.view([1, self.insize, 1, 1]).expand_as(x)
            if self.non_linear:
                den = torch.max(sigma, self.v * x + instance_std(x))
                return x / den * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta
