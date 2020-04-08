import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import uniform


def instance_std(x, eps=1e-5):
    var = torch.std(x, dim = (2, 3), keepdim=True)
    return torch.sqrt(var + eps)


def group_std(x, groups = 32, eps = 1e-5):
    N, C, H, W = x.size()
    x = torch.reshape(x, (N, groups, C // groups, H, W))
    var = torch.std(x, dim = (3, 4, 2), keepdim = True)
    return torch.reshape(torch.sqrt(var + eps), (N, C, H, W))


class EvoNorm2D(nn.Module):

    def __init__(self, input, non_linear = True, version = 'S0', training = False):
        super(EvoNorm2D, self).__init__()
        self.non_linear = non_linear
        self.version = version
        self.insize = input
        #print(self.insize)
        self.training = training
        if self.version not in ['B0', 'S0']:
            raise ValueError("Invalid EvoNorm version")
        U = uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        self.gamma = nn.Parameter(U.sample(torch.Size([self.insize])).view(self.insize))
        self.beta = nn.Parameter(torch.zeros(self.insize))
        if self.non_linear:
            self.v = nn.Parameter(torch.ones(1,self.insize,1,1))
        

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
            if self.non_linear:
                _, C, _, _ = x.size()
                tmp = x.permute(1,0,2,3).reshape(C, -1)
                sigma = tmp.std(dim=1).reshape(1,C,1,1)
                # sigma = x.std(dim=(0,2,3), keepdim=True)     #For Nightly Build only
                den = torch.max(sigma, self.v * x + instance_std(x))
                return x / den * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta
