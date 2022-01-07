import numpy as np
import torch
from torch import nn

import model, util, zernike
from . import base, encoder, renderer


class BaseMapperModel(base.BaseModel):
    
    def __init__(self, img_size, fit_params, max_psf_count, params_ref, params_ref_no_scale):
        super().__init__()
        
        self.setup_params_ref(img_size, fit_params, max_psf_count, params_ref, params_ref_no_scale)
        
        self.img_size = img_size
        self.mappings = list()
        self.mapping_modules = nn.ModuleDict()
        self.mapped_params = dict()
        self.in_channels = 0
        self.setup_mappings(fit_params, max_psf_count)
        
        self.cached_images = dict()
    
    def setup_params_ref(self, img_size, fit_params, max_psf_count, params_ref, params_ref_no_scale):
        if params_ref_no_scale is True:
            params_ref = self.generate_params_ref_no_scale(params_ref)
        
        params_ref_sorted = self.generate_params_ref_sorted(params_ref, fit_params)
        
        self.params_ref = params_ref_sorted
            
    def generate_params_ref_no_scale(self, params_ref):
        new_params_ref = dict()
        for key, val in params_ref.items():
            new_params_ref[key] = model.FitParameter(nn.Identity(), 0, 1,
                                                     (val.default + val.offset) * val.scaling,
                                                     val.per_psf)
        return new_params_ref
    
    def generate_params_ref_sorted(self, params_ref, fit_params):
        new_params_ref = dict()
        for param in fit_params:
            if not param in params_ref:
                raise Exception("fit param ({}) not recognised for this model".format(param))
            new_params_ref[param] = params_ref[param]
        for key, val in params_ref.items():
            if not key in new_params_ref:
                new_params_ref[key] = val
        return new_params_ref
    
    def setup_mappings(self, fit_params, max_psf_count):
        raise NotImplementedError()
    
    def forward(self, x):
        i = 0
        for in_channel_counts, models, (out_channel_names, out_channel_counts) in self.mappings:
            data = x[:, i:i+in_channel_counts, :, :]
            i += in_channel_counts
            
            data = models(data)
            
            data = data.split(out_channel_counts, dim=1)
            for j, out_channel_name in enumerate(out_channel_names):
                self.mapped_params[out_channel_name] = data[j]
                
        return self.mapped_params
    
    def get_suppl(self, colored=False):
        ret = dict()
        if len(self.cached_images) > 0:
            images = {}
            for key, val in self.cached_images.items():
                images[key] = util.color_images(util.tile_images(val[:9], 3)[0], full_output=True)
            ret['images'] = images
        return ret
    
    def get_random_mapped_params(self, batch_size):
        # x = (torch.rand((batch_size, self.in_channels, self.img_size[0], self.img_size[1])) - 0.5) * 2
        x = (torch.rand((batch_size, self.in_channels, 1, 1)) - 0.5) * 2
        x = self.forward(x)
        return x


class DirectMapperModel(BaseMapperModel):
    
    def __init__(self, img_size, fit_params, max_psf_count, new_params_ref, params_ref_no_scale):
        
        params_ref = self.generate_params_ref(new_params_ref, img_size)
        
        super().__init__(img_size=img_size, fit_params=fit_params,
                         max_psf_count=max_psf_count, params_ref=params_ref,
                         params_ref_no_scale=params_ref_no_scale)
        
    def generate_params_ref(self, new_params_ref, img_size):
        params_ref = dict()
        params_ref.update({
            'x': model.FitParameter(nn.Tanh(), 0, 0.75 * img_size[0], 0, True),
            'y': model.FitParameter(nn.Tanh(), 0, 0.75 * img_size[1], 0, True),
            'z': model.FitParameter(nn.Tanh(), 0, 2 * np.pi, 0, True),
            'A': model.FitParameter(nn.ReLU(), 0, 1000, 1, True),
            'bg': model.FitParameter(nn.Tanh(), 0, 500, 0, False),
            'p': model.FitParameter(nn.Sigmoid(), 0, 1, 1, True)
        })
        params_ref.update(new_params_ref)
        return params_ref
    
    def setup_mappings(self, fit_params, max_psf_count):
        for param_key, param_val in self.params_ref.items():
            repeats = 0
            if param_key in fit_params:
                repeats = max_psf_count if param_val.per_psf else 1
                self.mappings.append((repeats,
                                     nn.Sequential(param_val.activation, OffsetScale(param_val.offset, param_val.scaling)),
                                     [[param_key,], [repeats,],]))
            else:
                self.mapped_params[param_key] = torch.as_tensor(param_val.default, dtype=torch.float32)
            self.in_channels += repeats


class CentroidMapperModel(BaseMapperModel):
    
    def __init__(self, img_size, fit_params, max_psf_count, new_params_ref, params_ref_no_scale):
        
        params_ref = self.generate_params_ref(new_params_ref)
        
        super().__init__(img_size=img_size, fit_params=fit_params,
                         max_psf_count=max_psf_count, params_ref=params_ref,
                         params_ref_no_scale=params_ref_no_scale)
        
    def generate_params_ref(self, new_params_ref):
        params_ref = dict()
        params_ref.update({
            'x': model.FitParameter(nn.Identity(), 0, 1, 0, True),
            'y': model.FitParameter(nn.Identity(), 0, 1, 0, True),
            # 'z': model.FitParameter(nn.Tanh(), 0, 2 * np.pi, 0, True),
            'A': model.FitParameter(nn.ReLU(), 0, 1, 1, True),
            'bg': model.FitParameter(nn.Identity(), 0, 1, 0, False),
            'p': model.FitParameter(nn.Sigmoid(), 0, 1, 1, True)
        })
        params_ref.update(new_params_ref)
        return params_ref
    
    def setup_mappings(self, fit_params, max_psf_count):
        if any(param in fit_params for param in ['A', 'x', 'y']):
            centroid_module = Centroid(self.img_size)
            centroid_module.register_forward_hook(self.save_input_image)
            self.mappings.append((max_psf_count,
                                  nn.Sequential(nn.ReLU(),
                                                centroid_module),
                                  [['A', 'x', 'y'], [max_psf_count,]*3]))
            self.in_channels += max_psf_count
        for param_key, param_val in self.params_ref.items():
            repeats = 0
            if param_key in fit_params:
                if not any(p in param_key for p in ['A', 'x', 'y']):
                    repeats = max_psf_count if param_val.per_psf else 1
                    self.mappings.append((repeats,
                                         nn.Sequential(param_val.activation,
                                                       OffsetScale(param_val.offset, param_val.scaling),
                                                       Average()),
                                         [[param_key,], [repeats,]]))
            else:
                self.mapped_params[param_key] = torch.as_tensor(param_val.default, dtype=torch.float32)
            self.in_channels += repeats
            
    def save_input_image(self, module, input, output):
        self.cached_images["preCentroid"] = input[0][0,:9,...].detach()
            


class StackedConvMapper(BaseMapperModel):
    
    def __init__(self, img_size, fit_params, max_psf_count, new_params_ref, params_ref_no_scale):
        
        params_ref = self.generate_params_ref(new_params_ref, img_size)
        
        super().__init__(img_size=img_size, fit_params=fit_params,
                         max_psf_count=max_psf_count, params_ref=params_ref,
                         params_ref_no_scale=params_ref_no_scale)
        
    def generate_params_ref(self, new_params_ref, img_size):
        params_ref = dict()
        params_ref.update({
            'x': model.FitParameter(nn.Hardtanh(), 0, 1 * img_size[0], 0, True),
            'y': model.FitParameter(nn.Hardtanh(), 0, 1 * img_size[1], 0, True),
            # 'z': model.FitParameter(nn.Tanh(), 0, 2 * np.pi, 0, True),
            'A': model.FitParameter(nn.ReLU(), 0, 10000, 1, True),
            'bg': model.FitParameter(nn.Identity(), 0, 100, 0, False),
            'p': model.FitParameter(nn.Sigmoid(), 0, 1, 1, True)
        })
        params_ref.update(new_params_ref)
        return params_ref
    
    def setup_mappings(self, fit_params, max_psf_count):
        per_channel_params = list()
        per_channel_act_scale = list()
        for param_key, param_val in self.params_ref.items():
            repeats = 0
            if param_key in fit_params:
                if param_val.per_psf:
                    per_channel_params.append(param_key)
                    per_channel_act_scale.append(nn.Sequential(param_val.activation,
                                                               OffsetScale(param_val.offset, param_val.scaling),
                                                              ))
                else:
                    repeats = 1
                    self.mappings.append((repeats,
                                         nn.Sequential(param_val.activation,
                                                       OffsetScale(param_val.offset, param_val.scaling),
                                                       Average()),
                                         [[param_key,], [repeats,]]))
            else:
                self.mapped_params[param_key] = torch.as_tensor(param_val.default, dtype=torch.float32)
            self.in_channels += repeats
        
        in_channels = 8 * len(per_channel_params)
        out_channels = len(per_channel_params)
        self.mapping_modules['psf_params'] = nn.Sequential(#nn.ReLU(),
                                             StackConv(in_channels, out_channels,
                                                       max_psf_count,
                                                       dict(zip(per_channel_params, per_channel_act_scale))))
        self.mappings.append((in_channels * max_psf_count,
                              self.mapping_modules['psf_params'],
                              [per_channel_params, [max_psf_count,]*out_channels]))
        self.in_channels += in_channels * max_psf_count


class OffsetScale(nn.Module):
    def __init__(self, offset, scale):
        super().__init__()
        self.register_buffer("offset", torch.tensor(offset))
        self.register_buffer("scale", torch.tensor(scale))
    
    def forward(self, x):
        x = x + self.offset
        x = x * self.scale
        return x


class Centroid(nn.Module):
    def __init__(self, img_size, scale=[1,1]):
        super().__init__()
        xs = torch.arange(0, img_size[0], dtype=torch.float) * scale[0]
        xs -= xs.mean()
        ys = torch.arange(0, img_size[1], dtype=torch.float) * scale[1]
        ys -= ys.mean()
        XS, YS = torch.meshgrid(xs, ys, indexing='ij')
        XS = XS.unsqueeze(0).unsqueeze(0)
        YS = YS.unsqueeze(0).unsqueeze(0)
        
        AS = torch.ones(XS.shape)
        AXYS = torch.stack((AS, XS, YS), dim=1)
        
        self.register_buffer('AXYS', AXYS)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        
        x_sum = x.sum(dim=(-2, -1), keepdims=True)
        x_sum = x_sum.tile(1, 3, 1, 1, 1)
        x_sum = x_sum.clamp(min=1e-9)
        x_sum[:, 0] = 1
        
        x = x * self.AXYS
        x = x.sum(dim=(-2,-1), keepdims=True) / x_sum
        x = torch.cat([x[:,i,...] for i in range(x.shape[1])], dim=1)
        return x


class Average(nn.Module):
    def forward(self, x):
        x = x.mean(dim=(-2,-1), keepdims=True)
        return x


class StackConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_chunks, per_channel_act_scale):
        super().__init__()
        self.n_chunks = n_chunks
        self.encoder = self._encode_block(in_channels , out_channels)
        self.activation = nn.ModuleDict(per_channel_act_scale)
        
    def _encode_block(self, in_channels, out_channels):
        return nn.Sequential(
            # nn.GroupNorm(in_channels, in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.GELU(),
            # nn.GroupNorm(out_channels, out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            # nn.Dropout2d(0.5),
            nn.GELU(),
        )
    
    def forward(self, x):
        x = x.chunk(self.n_chunks, dim=1)
        x = torch.cat(x, dim=0)
        x = self.encoder(x)
        x = torch.cat([item(x[:,[i],...]) for i, item in enumerate(self.activation.values())], dim=1)
        x = x.chunk(self.n_chunks, dim=0)
        dim_len = len(x)
        x = torch.cat([x[i%dim_len][:,[i//dim_len],...] for i in range(len(x)*x[0].shape[1])], dim=1)
        return x