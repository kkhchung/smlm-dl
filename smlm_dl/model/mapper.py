import numpy as np
import torch
from torch import nn

import model, util, zernike
from . import base, encoder, renderer


class BaseMapperModel(base.BaseModel):
    
    def __init__(self):
        super().__init__()
    
    # def forward(self):
    #     super().forward()
    
    
class DirectMapperModel(BaseMapperModel):
    
    def __init__(self, img_size, fit_params, max_psf_count, new_params_ref, params_ref_no_scale):
        super().__init__()
        self.setup_fit_params(img_size, fit_params, max_psf_count, new_params_ref, params_ref_no_scale)
    
    def setup_fit_params(self, img_size, fit_params, max_psf_count, new_params_ref, params_ref_no_scale):
        self.params_ref = dict()
        self.params_ref.update({
            'x': model.FitParameter(nn.Tanh(), 0, 0.75 * img_size[0], 0, True),
            'y': model.FitParameter(nn.Tanh(), 0, 0.75 * img_size[1], 0, True),
            'z': model.FitParameter(nn.Tanh(), 0, 2 * np.pi, 0, True),
            'A': model.FitParameter(nn.ReLU(), 0, 1000, 1, True),
            'bg': model.FitParameter(nn.Tanh(), 0, 500, 0, False),
            'p': model.FitParameter(nn.Sigmoid(), 0, 1, 1, True)
        })
        self.params_ref.update(new_params_ref)
        
        if params_ref_no_scale is True:
            for key in self.params_ref:
                self.params_ref[key] = FitParameter(nn.Identity(), 0, 1,
                                                    (self.params_ref[key].default + self.params_ref[key].offset) * self.params_ref[key].scaling,
                                                    self.params_ref[key].per_psf)

        detailed_fit_params = dict()
        params_ref_sorted = dict()
        n_fit_params = 0
        for param in fit_params:
            if not param in self.params_ref:
                raise Exception("fit param ({}) not recognised for this model".format(param))
            repeats = max_psf_count if self.params_ref[param].per_psf else 1
            detailed_fit_params[param] = list()
            for i in range(repeats):
                detailed_fit_params[param].append(self.params_ref[param].copy())
                n_fit_params += 1
            params_ref_sorted[param] = self.params_ref.pop(param)
        params_ref_sorted.update(self.params_ref)
        self.params_ref.clear() 
        self.params_ref.update(params_ref_sorted)
        self.fit_params = detailed_fit_params
        # print(self.fit_params)
        self.n_fit_params = n_fit_params
        
    def forward(self, x):
        mapped_params = dict()
        i = 0
        for param in self.params_ref:
            if not param in self.fit_params:
                mapped_params[param] = torch.as_tensor(self.params_ref[param].default)
            else:
                repeats = len(self.fit_params[param])
                temp_params = x[:, i:i+repeats, ...]
                mapped_params[param] = torch.empty_like(temp_params)
                
                for j in range(temp_params.shape[1]):
                    mapped_params[param][:, j] = self.fit_params[param][j].activation(temp_params[:, j])
                    mapped_params[param][:, j] = torch.add(mapped_params[param][:, j], self.fit_params[param][j].offset)
                    mapped_params[param][:, j] = torch.mul(mapped_params[param][:, j], self.fit_params[param][j].scaling)
                
                i += repeats
                
        # print(mapped_params)
        return mapped_params
    
    
class CentroidMapperModel(BaseMapperModel):
    
    def __init__(self, img_size, fit_params, max_psf_count, new_params_ref, params_ref_no_scale):
        super().__init__()
        self.setup_fit_params(img_size, fit_params, max_psf_count, new_params_ref, params_ref_no_scale)
        
        self._set_buffers(img_size)
        self.cached_images = dict()
        
    def setup_fit_params(self, img_size, fit_params, max_psf_count, new_params_ref, params_ref_no_scale):
        self.params_ref = dict()
        self.params_ref.update({
            'x': model.FitParameter(nn.Identity(), 0, 1, 0, True),
            'y': model.FitParameter(nn.Identity(), 0, 1, 0, True),
            # 'z': model.FitParameter(nn.Tanh(), 0, 2 * np.pi, 0, True),
            'A': model.FitParameter(nn.ReLU(), 0, 1, 1, True),
            'bg': model.FitParameter(nn.Identity(), 0, 1, 0, False),
            'p': model.FitParameter(nn.Sigmoid(), 0, 1, 1, True)
        })
        self.params_ref.update(new_params_ref)
        
        if params_ref_no_scale is True:
            for key in self.params_ref:
                self.params_ref[key] = FitParameter(nn.Identity(), 0, 1,
                                                    (self.params_ref[key].default + self.params_ref[key].offset) * self.params_ref[key].scaling,
                                                    self.params_ref[key].per_psf)

        detailed_fit_params = dict()
        params_ref_sorted = dict()
        n_fit_params = 0
        for param in fit_params:
            if not param in self.params_ref:
                raise Exception("fit param ({}) not recognised for this model".format(param))
            repeats = max_psf_count if self.params_ref[param].per_psf else 1
            detailed_fit_params[param] = list()
            for i in range(repeats):
                detailed_fit_params[param].append(self.params_ref[param].copy())
                n_fit_params += 1
            params_ref_sorted[param] = self.params_ref.pop(param)
        params_ref_sorted.update(self.params_ref)
        self.params_ref.clear() 
        self.params_ref.update(params_ref_sorted)
        self.fit_params = detailed_fit_params
        self.n_fit_params = n_fit_params
        
    def forward(self, x):
        i = 0
        extracted_params = self._extract_params(x[:,:10])
        i += 10
        mapped_params = dict()
        for param in self.params_ref:
            if not param in self.fit_params:
                mapped_params[param] = torch.as_tensor(self.params_ref[param].default)
            else:
                repeats = len(self.fit_params[param])
                temp_params = x[:, i:i+repeats, ...]
                mapped_params[param] = torch.empty(temp_params.shape[:2] + (1,)*2)
                
                for j in range(temp_params.shape[1]):
                    if param in extracted_params:
                        mapped_params[param][:, j, 0, 0] = extracted_params[param][:, j]
                    else:
                        i += repeats
                        mapped_params[param][:, j] = self.fit_params[param][j].activation(temp_params[:, j]).sum(dim=(-2,-1), keepdims=True)
                    mapped_params[param][:, j] = torch.add(mapped_params[param][:, j], self.fit_params[param][j].offset)
                    mapped_params[param][:, j] = torch.mul(mapped_params[param][:, j], self.fit_params[param][j].scaling)
                
        return mapped_params
    
    def _set_buffers(self, img_size):
        xs = torch.arange(0, img_size[0], dtype=torch.float)
        xs -= xs.mean()
        ys = torch.arange(0, img_size[1], dtype=torch.float)
        ys -= ys.mean()
        
        XS, YS = torch.meshgrid(xs, ys, indexing='ij')
        XYS = torch.stack((XS, YS), dim=0).reshape(1, 1, 2, img_size[0], img_size[1])
        
        self.register_buffer('XYS', XYS)
        
    def _extract_params(self, x):
        self.cached_images['labelled'] = x.detach()[0,:]
        mapped_params = dict()
        mapped_params['A'] = x.sum(dim=(-2,-1))
        x = x.unsqueeze(2)
        x = x - x.amin(dim=(-2,-1), keepdims=True) + 1e-6
        x_weighted = x * self.XYS
        params = x_weighted.sum(dim=(-2,-1)) / x.sum(dim=(-2,-1))
        mapped_params['x'] = params[:, :, 0]
        mapped_params['y'] = params[:, :, 1]
        return mapped_params
    
    def get_suppl(self, colored=False):
        images = {}
        for key, val in self.cached_images.items():
            images['{}'.format(key)] = util.color_images(util.tile_images(val[:9], 3)[0], full_output=True)
                
        ret = super().get_suppl(colored=colored)
        if not 'images' in ret:
            ret['images'] = images
        else:
            ret['images'].update(images)
        return ret