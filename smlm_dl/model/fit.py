import functools
import numpy as np
import torch
from torch import nn

import util, zernike
from . import base, encoder, renderer


class FitParameter(object):
    activation = None
    offset = None
    scaling = None
    per_psf = None
    
    def __init__(self, activation, offset, scaling, default, per_psf):
        self.activation = activation
        self.offset = offset
        self.scaling = scaling
        self.default = default
        self.per_psf = per_psf
        
    def copy(self):
        return FitParameter(self.activation, self.offset, self.scaling, self.default, self.per_psf)
    
    def __repr__(self):
        text = "<"
        text += "act: {}, ".format(self.activation.__class__.__name__)
        text += "offset: {}, ".format(self.offset)
        text += "scaling: {}, ".format(self.scaling)
        text += "default: {}, ".format(self.default)
        text += "per psf: {}, ".format(self.per_psf)
        text += ">"
        return text


class BaseFitModel(base.BaseModel):
    
    def __init__(self, renderer_class, encoder_class, feedback_class=None,
                 img_size=(32,32), fit_params=['x', 'y', ], max_psf_count=1,
                 params_ref_override={}, params_ref_no_scale=False,
                 encoder_params={}, renderer_params={}, feedback_params={},):
        super().__init__()
        self.img_size = img_size
        self.setup_fit_params(img_size, fit_params, max_psf_count, params_ref_override, params_ref_no_scale)
        self.renderer = renderer_class(self.img_size, self.fit_params, **renderer_params)
        in_channels = 1
        if feedback_class is None:
            self.feedbacker = None
        else:
            feedback = self.renderer.get_feedback()
            self.feedbacker = feedback_class(img_size, feedback.shape[-2:], **feedback_params)
            in_channels += feedback.shape[1]
        if issubclass(encoder_class, encoder.ImageEncoderModel):
            encoder_params.update({"img_size":img_size, "in_channels":in_channels})
        self.encoder = encoder_class(last_out_channels=self.n_fit_params, **encoder_params)
        self.image_input = self.encoder.image_input
        
    def forward(self, x):
        if not self.feedbacker is None:
            x = self.feedbacker(x, self.renderer.get_feedback())
        x = self.encoder(x)
        batch_size = x.shape[0]
        x = self.map_params(x)
        self.mapped_params = x
        x = self.renderer(x, batch_size)
        return x
    
    def setup_fit_params(self, img_size, fit_params, max_psf_count, new_params_ref, params_ref_no_scale):
        self.params_ref = dict()
        self.params_ref.update({
            'x': FitParameter(nn.Tanh(), 0, 0.75 * img_size[0], 0, True),
            'y': FitParameter(nn.Tanh(), 0, 0.75 * img_size[1], 0, True),
            'z': FitParameter(nn.Tanh(), 0, 2 * np.pi, 0, True),
            'A': FitParameter(nn.ReLU(), 0, 1000, 1, True),
            'bg': FitParameter(nn.Tanh(), 0, 500, 0, False),
            'p': FitParameter(nn.Sigmoid(), 0, 1, 1, True)
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
        
    def map_params(self, x):
        mapped_params = dict()
        i = 0
        for param in self.params_ref:
            if not param in self.fit_params:
                mapped_params[param] = self.params_ref[param].default
            else:
                repeats = len(self.fit_params[param])
                temp_params = x[:, i:i+repeats, ...]
                mapped_params[param] = torch.empty_like(temp_params)
                
                for j in range(temp_params.shape[1]):
                    mapped_params[param][:, j] = self.fit_params[param][j].activation(temp_params[:, j])
                    mapped_params[param][:, j] = torch.add(mapped_params[param][:, j], self.fit_params[param][j].offset)
                    mapped_params[param][:, j] = torch.mul(mapped_params[param][:, j], self.fit_params[param][j].scaling)
                
                i += repeats
        return mapped_params
    
    def render_example_images(self, num):
        mapped_params = dict()
        i = 0
        for param in self.params_ref:
            if not param in self.fit_params:
                mapped_params[param] = self.params_ref[param].default
            else:
                repeats = len(self.fit_params[param])
                mapped_params[param] = torch.rand((num, repeats, 1, 1))
                
                for j in range(mapped_params[param].shape[1]):
                    if isinstance(self.fit_params[param][j].activation, (nn.Tanh, nn.Hardtanh)):
                        mapped_params[param][:, j] = (mapped_params[param][:, j] - 0.5) * 2
                    elif isinstance(self.fit_params[param][j].activation, (nn.Identity, nn.ReLU, nn.Sigmoid)):
                        mapped_params[param][:, j] = mapped_params[param][:, j]
                    else:
                        raise Exception("Unrecognized activation function. [{}]".format(type(self.fit_params[param][j].activation)))
                    mapped_params[param][:, j] = self.fit_params[param][j].activation(mapped_params[param][:, j])
                    mapped_params[param][:, j] = torch.add(mapped_params[param][:, j], self.fit_params[param][j].offset)
                    mapped_params[param][:, j] = torch.mul(mapped_params[param][:, j], self.fit_params[param][j].scaling)
                
                i += repeats
        # print(mapped_params)
        images = self.renderer.render_images(mapped_params, num, True)
        return images
    
    def get_suppl(self, colored=False):
        ret = dict()
        if hasattr(self.renderer, 'get_suppl'):
            ret.update(self.renderer.get_suppl(colored=colored))
        return ret


class UnetFitModel(BaseFitModel):
    def __init__(self, renderer_class, encoder_class, feedback_class=None,
                 img_size=(32,32), fit_params=['x', 'y', ], max_psf_count=1,
                 params_ref_override={}, params_ref_no_scale=False,
                 encoder_params={}, renderer_params={}, feedback_params={},):
        
        super().__init__(renderer_class=renderer_class, encoder_class=encoder_class, feedback_class=feedback_class,
                 img_size=img_size, fit_params=fit_params, max_psf_count=max_psf_count,
                 params_ref_override=params_ref_override, params_ref_no_scale=params_ref_no_scale,
                 encoder_params=encoder_params, renderer_params=renderer_params, feedback_params=feedback_params,)
        
        self._set_buffers()
        self.cached_images = dict()
    
    def setup_fit_params(self, img_size, fit_params, max_psf_count, new_params_ref, params_ref_no_scale):
        self.params_ref = dict()
        self.params_ref.update({
            'x': FitParameter(nn.Identity(), 0, 1, 0, True),
            'y': FitParameter(nn.Identity(), 0, 1, 0, True),
            # 'z': FitParameter(nn.Tanh(), 0, 2 * np.pi, 0, True),
            'A': FitParameter(nn.ReLU(), 0, 1, 1, True),
            'bg': FitParameter(nn.Identity(), 0, 1, 0, False),
            'p': FitParameter(nn.Sigmoid(), 0, 1, 1, True)
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
        
    def map_params(self, x):
        i = 0
        extracted_params = self._extract_params(x[:,:10])
        i += 10
        mapped_params = dict()
        for param in self.params_ref:
            if not param in self.fit_params:
                mapped_params[param] = self.params_ref[param].default
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
    
    def _set_buffers(self):
        xs = torch.arange(0, self.img_size[0], dtype=torch.float)
        xs -= xs.mean()
        ys = torch.arange(0, self.img_size[1], dtype=torch.float)
        ys -= ys.mean()
        
        XS, YS = torch.meshgrid(xs, ys, indexing='ij')
        XYS = torch.stack((XS, YS), dim=0).reshape(1, 1, 2, self.img_size[0], self.img_size[1])
        
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


class Gaussian2DModel(BaseFitModel):
    def __init__(self, fit_params=['x', 'y', 'sig'], encoder_class=encoder.ConvImageEncoderModel, **kwargs):
        super().__init__(encoder_class=encoder_class, renderer_class=renderer.Gaussian2DRenderer, fit_params=fit_params, **kwargs)
    
    def setup_fit_params(self, img_size, fit_params, max_psf_count, new_params_ref, params_ref_no_scale):
        params_ref = {
            'sig': FitParameter(nn.ReLU(), 2, 1, 5, True),
        }
        params_ref.update(new_params_ref)
        
        BaseFitModel.setup_fit_params(self, img_size, fit_params, max_psf_count, params_ref, params_ref_no_scale)


class Template2DModel(BaseFitModel):
    def __init__(self, fit_params=['x', 'y'], encoder_class=encoder.ConvImageEncoderModel, **kwargs):
        super().__init__(encoder_class=encoder_class, renderer_class=renderer.Template2DRenderer, fit_params=fit_params,
                              **kwargs)


class FourierOptics2DModel(BaseFitModel):
    def __init__(self, fit_params=['x', 'y'], encoder_class=encoder.ConvImageEncoderModel, **kwargs):
        super().__init__(encoder_class=encoder_class, renderer_class=renderer.FourierOptics2DRenderer, fit_params=fit_params, **kwargs)