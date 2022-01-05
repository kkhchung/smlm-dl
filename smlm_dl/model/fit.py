import functools
import numpy as np
import torch
from torch import nn

import model, util, zernike
from . import base, encoder, mapper, renderer


class BaseFitModel(base.BaseModel):
    
    def __init__(self, renderer_class, encoder_class,
                 mapper_class=mapper.DirectMapperModel, feedback_class=None,
                 img_size=(32,32), fit_params=['x', 'y', ], max_psf_count=1,
                 params_ref_override={}, params_ref_no_scale=False,
                 encoder_params={}, renderer_params={}, feedback_params={},):
        super().__init__()
        self.img_size = img_size
        self.mapper = mapper_class(img_size, fit_params, max_psf_count, params_ref_override, params_ref_no_scale)
        self.renderer = renderer_class(self.img_size, self.mapper.fit_params, **renderer_params)
        in_channels = 1
        if feedback_class is None:
            self.feedbacker = None
        else:
            feedback = self.renderer.get_feedback()
            self.feedbacker = feedback_class(img_size, feedback.shape[-2:], **feedback_params)
            in_channels += feedback.shape[1]
        if issubclass(encoder_class, encoder.ImageEncoderModel):
            encoder_params.update({"img_size":img_size, "in_channels":in_channels})
        self.encoder = encoder_class(last_out_channels=self.mapper.n_fit_params, **encoder_params)
        self.image_input = self.encoder.image_input
        
    def forward(self, x):
        if not self.feedbacker is None:
            x = self.feedbacker(x, self.renderer.get_feedback())
        x = self.encoder(x)
        batch_size = x.shape[0]
        x = self.mapper(x)
        self.mapped_params = x
        x = self.renderer(x, batch_size)
        return x
    
    def render_example_images(self, num):
        mapped_params = dict()
        i = 0
        for param in self.mapper.params_ref:
            if not param in self.fit_params:
                mapped_params[param] = self.mapper.params_ref[param].default
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
        for key, val in {'E':self.encoder, 'M':self.mapper,
                         'R':self.renderer, 'F':self.feedbacker}.items():
            if hasattr(val, 'get_suppl'):
                suppls = val.get_suppl(colored=colored)
                for suppl_type_key, suppl_type_val in suppls.items():
                    if not suppl_type_key in ret:
                        ret[suppl_type_key] = dict()
                    for suppl_key, suppl_val in suppl_type_val.items():
                        ret[suppl_type_key]["{}:{}".format(key, suppl_key)] = suppl_val
        return ret
    
    @property
    def fit_params(self):
        return self.mapper.fit_params


class Gaussian2DModel(BaseFitModel):
    def __init__(self, fit_params=['x', 'y', 'sig'], encoder_class=encoder.ConvImageEncoderModel, **kwargs):
        params_ref = {
            'sig': model.FitParameter(nn.ReLU(), 2, 1, 5, True),
        }
        params_ref.update(kwargs.get('params_ref_override', {}))
        super().__init__(encoder_class=encoder_class, renderer_class=renderer.Gaussian2DRenderer,
                         fit_params=fit_params, params_ref_override=params_ref,
                         **kwargs)


class Template2DModel(BaseFitModel):
    def __init__(self, fit_params=['x', 'y'], encoder_class=encoder.ConvImageEncoderModel, **kwargs):
        super().__init__(encoder_class=encoder_class, renderer_class=renderer.Template2DRenderer, fit_params=fit_params,
                              **kwargs)


class FourierOptics2DModel(BaseFitModel):
    def __init__(self, fit_params=['x', 'y'], encoder_class=encoder.ConvImageEncoderModel, **kwargs):
        super().__init__(encoder_class=encoder_class, renderer_class=renderer.FourierOptics2DRenderer, fit_params=fit_params, **kwargs)