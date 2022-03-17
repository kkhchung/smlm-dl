from functools import partial

import torch
from torch import nn

class CameraLoss(nn.Module):
    def __init__(self, mle, gain=1, offset=100):
        super().__init__()
        self.mle = mle
        
        gain = nn.Parameter(torch.ones(1) * gain)
        offset = nn.Parameter(torch.ones(1) * offset)
        self.camera = nn.ParameterDict({
            'gain' : gain,
            'offset' : offset,
        })
        
    def forward(self, input, target):
        raise NotImplemented()
    
    def _to_photons(self, target):
        target = target - self.camera['offset']
        target = target / self.camera['gain']
        return target


class L2CameraLoss(CameraLoss):
    def __init__(self, gain=1, offset=100, **kwargs):
        super().__init__(mle=False, gain=gain, offset=offset)
    
    def forward(self, input, target, reduction='mean'):
        target = self._to_photons(target)
        loss = nn.functional.mse_loss(input, target, reduction=reduction)
        return loss


class L2PoissonWeightedCameraLoss(CameraLoss):
    def __init__(self, gain=1, offset=100, var=0):
        super().__init__(mle=False, gain=gain, offset=offset)
        self.var = var
    
    def forward(self, input, target, reduction='mean'):
        var = torch.clamp(input, 0) + self.var
        target = self._to_photons(target)
        loss = (target-input)**2 / var
        if reduction == 'mean':
            loss = torch.mean(loss)
        elif reduction == 'sum':
            loss = torch.sum(loss)
        else:
            raise Exception("reduction method not recognized")
        
        return loss


class GaussMLECameraLoss(CameraLoss):
    def __init__(self, gain=1, offset=100, var=0):
        super().__init__(mle=True, gain=gain, offset=offset)
        self.var = var
    
    def forward(self, input, target, reduction='mean'):
        target = self._to_photons(target)
        loss = nn.functional.gaussian_nll_loss(input, target, torch.ones_like(input)*self.var, reduction=reduction)
        return loss
    

class GaussMLEPoissonWeightedCameraLoss(CameraLoss):
    def __init__(self, gain=1, offset=100, var=0):
        super().__init__(mle=True, gain=gain, offset=offset)
        self.var = var
    
    def forward(self, input, target, reduction='mean'):
        var = torch.clamp(input, 0) + self.var
        target = self._to_photons(target)
        loss = nn.functional.gaussian_nll_loss(input, target, var, reduction=reduction)
        return loss


class GaussPoissonMLECameraLoss(CameraLoss):
    def __init__(self, gain=1, offset=100, var=0):
        super().__init__(mle=True, gain=gain, offset=offset)
        self.var = var
    
    def forward(self, input, target, reduction='mean'):
        target = self._to_photons(target)
        
        pos_diff = target - input
        input_poisson = torch.clamp(input, 0)
        poisson_var_proportion = torch.sqrt(input_poisson / (input_poisson + self.var))
        self.noise = nn.ParameterDict({
            'poisson' : nn.Parameter(torch.clamp(input_poisson + pos_diff * poisson_var_proportion, 0))
        })
        self.optimizer = torch.optim.LBFGS(self.noise.values(), lr=0.5,
                                      max_iter=10, tolerance_grad=1e-3,
                                     tolerance_change=1e-3,
                                           history_size=10)
            
        self.optimizer.step(partial(self._nested_closure, input=input.detach(), target=target.detach()))
        self.optimizer.zero_grad()
        loss = self._calculate_nll(input, target)
        
        return loss
    
    def _calculate_nll(self, input, target):
        eps = 1e-8
        poisson_input = torch.clamp(input, 0)
        target_with_poisson = self.noise['poisson'] - torch.clamp(self.noise['poisson'].detach(), None, 0)
        # the built-in Stirling approx is breaking autograd
        poisson_nll = nn.functional.poisson_nll_loss(poisson_input, target_with_poisson, log_input=False, full=False, eps=eps, reduction='mean')
        
        stirling_term = target_with_poisson * torch.log(target_with_poisson + eps) - target_with_poisson + 0.5 * torch.log(2 * torch.pi * target_with_poisson + eps)
        stirling_term[target_with_poisson<=1] = 0
        stirling_term = stirling_term.mean()
        poisson_nll = poisson_nll + stirling_term
        
        gaussian_noise = target - target_with_poisson + poisson_input
        gauss_nll = nn.functional.gaussian_nll_loss(input, gaussian_noise, torch.ones_like(gaussian_noise)*self.var, full=True, reduction='mean')
        
        return poisson_nll + gauss_nll
        
    def _nested_closure(self, input, target):
        self.optimizer.zero_grad()
        loss = self._calculate_nll(input, target)
        loss.backward()
        return loss