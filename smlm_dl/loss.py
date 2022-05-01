from functools import partial

import numpy as np

import torch
from torch import nn

from . import model

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
            # "base": model.base.ParameterModule(-torch.log(1/poisson_std_proportion-1)),
            # "clip": nn.Sigmoid(),
            "base": model.base.ParameterModule(6*(poisson_std_proportion-0.5)),
            "clip": nn.Hardsigmoid(),
        })
        self.optimizer = torch.optim.LBFGS(self.prop_poisson_noise["base"].parameters(),
                                           lr=1.0,
                                           max_iter=50,
                                           max_eval=100,
                                           tolerance_grad=1e-3,
                                           tolerance_change=1e-4,
                                           history_size=10,
                                           line_search_fn=None,
                                          )
        
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
    
    def _calculate_poisson_nll(self, input, target, eps=1e-8, factorial_appox="lgamma"):
        poisson_nll_a = input - target * torch.log(input + eps)
        if factorial_appox=="stirling":
            poisson_nll_b = target * torch.log(target+eps) - target + 0.5 * torch.log(2*torch.pi*target+eps)
            poisson_nll_b = torch.clamp(poisson_nll_b, 0)
            # poisson_nll_b[target <= 1] = 0
        elif factorial_appox=="lgamma":
            poisson_nll_b = torch.lgamma(target + 1)
        else:
            raise Exception("factorial_appox not recognised")
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


class SelfLimitingLoss(nn.Module):
    def __init__(self, crop=0, n_inner_iter=(300, 2), delay=100, loss_func=nn.MSELoss):
        super().__init__()
        
        self.crop = crop
        self.n_inner_iter_first = n_inner_iter[0]
        self.n_inner_iter_other = n_inner_iter[1]
        
        self.signal_model = model.encoder.UnetEncoderModel(img_size=(32,32), depth=4,
                                                           last_out_channels=1, act_func=nn.GELU,
                                                           final_act_func=nn.Identity)
        self.signal_loss_func = loss_func()
        self.signal_optimizer = torch.optim.AdamW(self.signal_model.parameters(), lr=1e-4)
        self.register_buffer('blank_ones', torch.ones((512, 1, 32, 32), device='cuda', requires_grad=False))
        
        self.noise_model = model.encoder.UnetEncoderModel(img_size=(32,32), depth=4,
                                                          last_out_channels=1, act_func=nn.GELU,
                                                          final_act_func=nn.Identity)
        self.noise_loss_func = loss_func()
        self.noise_optimizer = torch.optim.AdamW(self.noise_model.parameters(), lr=1e-4)
        
        self.loss_func = loss_func()
        
        self.error_network_on = -1
        self.n_outer_iter = 0
        self.n_outer_iter_delay = delay
        
        self.saved_scalars = dict()        
        
    def forward(self, input, target):        
        diff = target - input
        
        target_cropped = target[:,:,self.crop:-self.crop,self.crop:-self.crop]
        input_cropped = input[:,:,self.crop:-self.crop,self.crop:-self.crop]        
        
        loss = self.loss_func(target_cropped, input_cropped)
        
        if self.error_network_on >= 0:
            if self.training is True:
                signal_log, loss_log = list(), list()
                
                input_copy = input_cropped.detach()
                input_copy = input_copy - input_copy.mean()
                
                diff_copy = diff.detach()
                diff_copy = diff_copy[:,:,self.crop:-self.crop,self.crop:-self.crop]
                
                self.signal_model.train(True)                
                for i in range(self.n_inner_iter_first if self.error_network_on==0 else self.n_inner_iter_other):
                    self.signal_optimizer.zero_grad()
                    predict_signal = self.signal_model(self.blank_ones)
                    
                    predict_signal = predict_signal[:,:,self.crop:-self.crop,self.crop:-self.crop]                    
                    signal_predict_loss = self.signal_loss_func(input_copy,
                                                     predict_signal)

                    signal_predict_loss.backward()          
                    self.signal_optimizer.step()          
                    signal_log.append(signal_predict_loss.detach())
                    
                self.noise_model.train(True)                
                for i in range(self.n_inner_iter_first if self.error_network_on==0 else self.n_inner_iter_other):
                    self.noise_optimizer.zero_grad()
                    predict_diff = self.noise_model(input.detach())

                    predict_diff = predict_diff[:,:,self.crop:-self.crop,self.crop:-self.crop]
                    diff_predict_loss = self.noise_loss_func(diff_copy,
                                                     predict_diff)

                    diff_predict_loss.backward()                    
                    self.noise_optimizer.step()                    
                    loss_log.append(diff_predict_loss.detach())
                    
                self.error_network_on += 1            
            
            input_copy = input_cropped
            diff_copy = diff[:,:,self.crop:-self.crop,self.crop:-self.crop]
            target_copy = target_cropped
            target_copy_var = target_copy.var().detach()
            
            self.signal_model.train(False)
            predict_signal = self.signal_model(self.blank_ones)
            predict_signal = predict_signal[:,:,self.crop:-self.crop,self.crop:-self.crop]
            signal_predict_loss = self.signal_loss_func(input_copy - input_copy.mean(),
                                                        predict_signal)

            self.noise_model.train(False)
            predict_diff = self.noise_model(input)
            predict_diff = predict_diff[:,:,self.crop:-self.crop,self.crop:-self.crop]        
            diff_predict_loss = self.noise_loss_func(diff_copy,
                                                     predict_diff)

            signal_predict_loss = torch.clamp(signal_predict_loss, max=torch.mean(input_copy*input_copy).detach())
            diff_predict_loss = torch.clamp(diff_predict_loss, max=torch.mean(diff_copy*diff_copy).detach())
            
            grand_mean_input = input_copy.mean()**2
            grand_mean_target = target_copy.mean()**2
            signal_input = input_copy.var()
            
            self.saved_scalars["0_grand_mean_input"] = grand_mean_input
            self.saved_scalars["0_grand_mean_target"] = grand_mean_target
            self.saved_scalars["1_signal_input"] = signal_input
            self.saved_scalars["1_signal_predict"] = signal_predict_loss
            self.saved_scalars["2_loss"] = loss
            self.saved_scalars["2_loss_predict"] = diff_predict_loss
            
            total_loss = torch.abs(grand_mean_input - grand_mean_target)
            total_loss = total_loss + signal_input - torch.clamp(signal_predict_loss, max=target_copy_var)
            total_loss = total_loss + loss - torch.clamp(diff_predict_loss, max=target_copy_var)
            
        else:
            self.saved_scalars["0_grand_mean_input"] = torch.zeros(1)
            self.saved_scalars["0_grand_mean_target"] = torch.zeros(1)
            self.saved_scalars["1_signal_input"] = torch.zeros(1)
            self.saved_scalars["1_signal_predict"] = torch.zeros(1)
            self.saved_scalars["2_loss"] = loss
            self.saved_scalars["2_loss_predict"] = torch.zeros(1)
            
            total_loss = loss
        
        if self.n_outer_iter > self.n_outer_iter_delay and self.error_network_on == -1 and loss < target_cropped.var() * 0.5:
            self.error_network_on += 1
        
        self.n_outer_iter += 1
        
        return total_loss