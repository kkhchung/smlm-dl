from functools import partial

import numpy as np

import torch
from torch import nn

import model

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
        loss = nn.functional.gaussian_nll_loss(input, target, torch.ones_like(input)*self.var, full=True, reduction=reduction)
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
    def __init__(self, gain=1, offset=100, var=0, debug=False):
        super().__init__(mle=True, gain=gain, offset=offset)
        self.var = var
        self.debug = debug
        if self.debug:
            self.cached_images = dict()
            self.optim_counts = 0
    
    def forward(self, input, target, reduction='mean'):
        target = self._to_photons(target)
        
        pos_diff = target - input
        input_poisson = torch.clamp(input, 0)
        poisson_std_proportion = torch.sqrt(input_poisson / (input_poisson + self.var))
        self.prop_poisson_noise = nn.ModuleDict({
            "base": model.base.ParameterModule(-torch.log(1/poisson_std_proportion-1)),
            "clip": nn.Sigmoid(),
        })
        self.optimizer = torch.optim.LBFGS(self.prop_poisson_noise["base"].parameters(), lr=0.5,
                                           max_iter=50,
                                           tolerance_grad=1e-1,
                                           tolerance_change=1e-3,
                                           history_size=10)
        
        if self.debug:
            self.optim_counts = 0
        self.optimizer.step(partial(self._nested_closure, input=input.detach(), target=target.detach()))
        if self.debug:
            print(self.optim_counts)
        self.optimizer.zero_grad()
        loss = self._calculate_nll(input, target, round_poisson=True)
        
        return loss
    
    def _calculate_nll(self, input, target, round_poisson=False):
        eps = 1e-8
        poisson_input = torch.clamp(input, 0)

        prop_poisson_noise = None
        for module in self.prop_poisson_noise.values():
            prop_poisson_noise = module(prop_poisson_noise)
        error = target - input
        target_clipped = target - torch.clamp(target, None, 0)
        poisson_noise = target_clipped - input
        poisson_noise[input <= 0] = 0
        poisson_noise = poisson_noise * prop_poisson_noise
        if round_poisson is True:
            poisson_noise = torch.round(poisson_noise)
        
        poisson_nll = self._calculate_poisson_nll(poisson_input, poisson_input + poisson_noise)
        gaussian_nll = self._calculate_gaussian_nll(error - poisson_noise, self.var)
        nll = poisson_nll + gaussian_nll
        
        if self.debug:
            self.cached_images["input"] = input.detach()
            self.cached_images["target"] = target.detach()
            self.cached_images["poisson_noise"] = poisson_noise.detach()
            self.cached_images["gaussian_noise"] = (error - poisson_noise).detach()
            self.cached_images["nll"] = nll.detach()
            print('noises', poisson_noise[0,0,0,0], error[0,0,0,0] - poisson_noise[0,0,0,0], error[0,0,0,0])
            print('p', poisson_noise[0,0,0,0] / error[0,0,0,0])
        
        nll = nll.mean()
        
        return nll
        
    def _nested_closure(self, input, target):
        self.optimizer.zero_grad()
        loss = self._calculate_nll(input, target)
        loss.backward()
        if self.debug:
            self.optim_counts = self.optim_counts + 1
        return loss
    
    def _calculate_poisson_nll(self, input, target, eps=1e-8):
        poisson_nll_a = input - target * torch.log(input + eps) 
        # poisson_nll_b = target * np.log(target) - target + 0.5 * np.log(2*torch.pi*target)
        poisson_nll_b = torch.lgamma(target + 1)
        if self.debug:
            print('poisson nll', poisson_nll_a[0,0,0,0], poisson_nll_b[0,0,0,0])
            # print('poisson nll', poisson_nll_a.squeeze(), poisson_nll_b.squeeze(), poisson_nll_a.squeeze() + poisson_nll_b.squeeze())
            self.cached_images["poisson_nll_a"] = poisson_nll_a.detach()
            self.cached_images["poisson_nll_b"] = poisson_nll_b.detach()
        return poisson_nll_a + poisson_nll_b
    
    def _calculate_gaussian_nll(self, error, var):
        gaussian_nll_a = 0.5 * (np.log(var) + (error**2 / var))
        gaussian_nll_b = 0.5 * np.log(2*np.pi)
        if self.debug:
            print('gaussian nll', gaussian_nll_a[0,0,0,0], gaussian_nll_b)
            # print('gaussian nll', gaussian_nll_a.squeeze(), gaussian_nll_b.squeeze(), gaussian_nll_a.squeeze() + gaussian_nll_b.squeeze())
            self.cached_images["gaussian_nll_a"] = gaussian_nll_a.detach()
            self.cached_images["gaussian_nll_b"] = gaussian_nll_b
        return gaussian_nll_a + gaussian_nll_b