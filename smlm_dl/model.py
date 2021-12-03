import numpy as np

import torch
from torch import nn

# from torchinfo import summary

import matplotlib
from matplotlib import pyplot as plt

import warnings

import util

class EncoderModel(nn.Module):
    def __init__(self, img_size=(32,32), depth=3, first_layer_out_channels=16, last_out_channels=2, skip_channels=0, *args, **kwargs):
        nn.Module.__init__(self)
        if img_size[0] % 2 != 0 or img_size[1] % 2 !=0:
            raise Exception("Image input size needs to be multiples of two.")
        if 2**depth > img_size[0] or 2**depth > img_size[1]:
            raise Exception("Model too deep for this image size (depth = {}, image size needs to be at least ({}, {}))".format(depth, 2**depth, 2**depth))
            
        self.depth = depth
        self.img_size = img_size
        
        self.encoders = nn.ModuleDict()
        self.skips = nn.ModuleDict()
        
        for i in range(self.depth):
            in_channels = 1 if i == 0 else 2**(i-1) * first_layer_out_channels
            out_channels = 2**i * first_layer_out_channels
            
            if skip_channels > 0:
                in_shape = (int(img_size[0] * 0.5**i), int(img_size[1] * 0.5**i))
                self.skips["skip_conv_layer{}".format(i)] = self._skip_block(in_channels, skip_channels, in_shape)
            self.encoders["conv_layer{}".format(i)] = self._encode_block(in_channels, out_channels)
            
        self.neck = nn.ModuleDict()
        self.neck["conv_layer_0"] = self._neck_block(int(2**(self.depth-1) * first_layer_out_channels),
                                             (2**(self.depth) * first_layer_out_channels))
        self.neck["conv_layer_1"] = self._encode_final_block(2**(self.depth) * first_layer_out_channels,
                                                             2**(self.depth-1) * first_layer_out_channels,
                                                            tuple(int(size*0.5**self.depth) for size in self.img_size))
        
        self.decoders = nn.ModuleDict()
        self.decoders["dense_layer_0"] = nn.Conv2d(2**(self.depth-1) * first_layer_out_channels + skip_channels*self.depth, last_out_channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        skips = dict()
        
        if len(self.skips) > 0:        
            for i, ((encoder_key, encoder_module), (skip_key, skip_module)) in enumerate(zip(self.encoders.items(), self.skips.items())):
                skips[skip_key] = skip_module(x)
                x = encoder_module(x)
        else:
            for i, (key, module) in enumerate(self.encoders.items()):
                x = module(x)
        
        for i, (key, module) in enumerate(self.neck.items()):
            x = module(x)

        x = torch.cat([x] + list(skips.values()), dim=1)
        for i, (key, module) in enumerate(self.decoders.items()):
            x = module(x)
            
        return x
    
    def _skip_block(self, in_channels, out_channels, in_shape):
        return nn.Sequential(            
            nn.GroupNorm(in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=in_shape, padding=0),
            nn.ReLU(),
            nn.GroupNorm(out_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Dropout2d(0.5),
        )
    
    def _encode_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.GroupNorm(in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(out_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5),
        )
    
    def _neck_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.GroupNorm(in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(out_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.5),
        )
    
    def _encode_final_block(self, in_channels, out_channels, image_shape):
        return nn.Sequential(
            nn.GroupNorm(in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=image_shape, padding=0),
            nn.ReLU(),
        )
    

    
class BaseFitModel(EncoderModel):
    params_ref = dict()
    
    def __init__(self, renderer_class, img_size=(32,32), fit_params=['x', 'y', ], max_psf_count=1, params_ref_override={}, *args, **kwargs):
        self.setup_fit_params(img_size, fit_params, max_psf_count, params_ref_override)
        EncoderModel.__init__(self, img_size=img_size, last_out_channels=self.n_fit_params, *args, **kwargs)
        self.renderer = renderer_class(self.img_size, self.fit_params)
        
    def forward(self, x):
        x = EncoderModel.forward(self, x)
        batch_size = x.shape[0]
        x = self.map_params(x)
        self.mapped_params = x
        x = self.renderer(x, batch_size)
        return x
    
    def setup_fit_params(self, img_size, fit_params, max_psf_count, new_params_ref={},):
        self.params_ref.update({
            'x': FitParameter(nn.Tanh(), 0, 0.75 * img_size[0], 0, True),
            'y': FitParameter(nn.Tanh(), 0, 0.75 * img_size[1], 0, True),
            'z': FitParameter(nn.Tanh(), 0, 2 * np.pi, 0, True),
            'A': FitParameter(nn.ReLU(), 0, 1000, 1, True),
            'bg': FitParameter(nn.Tanh(), 0, 500, 0, False),
            'p': FitParameter(nn.Sigmoid(), 0, 1, 1, True)
        })        
        
        self.params_ref.update(new_params_ref)
        
        detailed_fit_params = dict()
        n_fit_params = 0
        for param in fit_params:
            if not param in self.params_ref:
                raise Exception("fit param ({}) not recognised for this model".format(param))
            repeats = max_psf_count if self.params_ref[param].per_psf else 1
            detailed_fit_params[param] = list()
            for i in range(repeats):
                detailed_fit_params[param].append(self.params_ref[param].copy())
                n_fit_params += 1
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
                mapped_params[param] = x[:, i:i+repeats, ...]
                
                for j in range(mapped_params[param].shape[1]):
                    mapped_params[param][:, j] = self.fit_params[param][j].activation(mapped_params[param][:, j])
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
                    if isinstance(self.fit_params[param][j].activation, nn.Tanh):
                        mapped_params[param][:, j] = (mapped_params[param][:, j] - 0.5) * 2
                    elif isinstance(self.fit_params[param][j].activation, nn.ReLU):
                        mapped_params[param][:, j] = mapped_params[param][:, j]
                    elif isinstance(self.fit_params[param][j].activation, nn.Sigmoid):
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
    
    
class BaseRendererModel(nn.Module):
    """
    Base renderer class for a few standardized output functions
    """
    params_ref = dict()
    params_default = dict()
    params_ind = dict()
    
    def __init__(self, img_size, fit_params):
        nn.Module.__init__(self)
        self.img_size = img_size
        self.fit_params = fit_params
        
    def forward(self, x, batch_size=None):
        return self.render_images(x, batch_size)
    
    def render_images(self, params, batch_size, as_numpy_array=False):
        images = self._render_images(params, batch_size)
        images = images.sum(dim=1, keepdim=True)        
        if as_numpy_array:
            images = images.detach().numpy()
        return images
        
    def _render_images(self, params, batch_size):
        # explicit call to render images without going through the rest of the model
        raise NotImplementedError()

        
class Gaussian2DModel(BaseFitModel):
    def __init__(self, fit_params=['x', 'y', 'sig'], *args, **kwargs):
        BaseFitModel.__init__(self, renderer_class=Gaussian2DRenderer, fit_params=fit_params, *args, **kwargs)
    
    def setup_fit_params(self, img_size, fit_params, max_psf_count, new_params_ref):
        params_ref = {
            'sig': FitParameter(nn.ReLU(), 2, 1, 5, True),
        }
        params_ref.update(new_params_ref)
        
        BaseFitModel.setup_fit_params(self, img_size, fit_params, max_psf_count, params_ref)

    
class Gaussian2DRenderer(BaseRendererModel):
    
    def __init__(self, img_size, fit_params):
        
        BaseRendererModel.__init__(self, img_size, fit_params)
        
        xs = torch.arange(0, self.img_size[0]) - 0.5*(self.img_size[0]-1)
        ys = torch.arange(0, self.img_size[1]) - 0.5*(self.img_size[1]-1)
        XS, YS = torch.meshgrid(xs, ys, indexing='ij')
        self.register_buffer('XS', XS, False)
        self.register_buffer('YS', YS, False)
    
    def _render_images(self, mapped_params, batch_size=None, ):
        
        images = torch.exp(-((self.XS[None,...]-mapped_params['x'])**2/mapped_params['sig'] \
                       + (self.YS[None,...]-mapped_params['y'])**2/mapped_params['sig']))
        
        images = images * mapped_params['A'] + mapped_params['bg']
        # print(mapped_params['x'][0], mapped_params['y'][0], mapped_params['p'])
        images = images * (mapped_params['p']>0.5)
        
        return images
    
    
class Template2DModel(BaseFitModel):
    def __init__(self, fit_params=['x', 'y'], *args, **kwargs):
        BaseFitModel.__init__(self, renderer_class=Template2DRenderer, fit_params=fit_params, *args, **kwargs)
    
    def get_suppl(self, colored=False):
        template = self.renderer._calculate_template(True).detach().numpy()
        template_2x = self.renderer._calculate_template(False).detach().numpy()
        if colored:
            template = util.color_images(template, full_output=True)
            template_2x = util.color_images(template_2x, full_output=True)
        return {'template':template, 'template 2x':template_2x, }

    
class Template2DRenderer(BaseRendererModel):
    def __init__(self, img_size, fit_params):
        # The 2x sampling avoids the banding artifact that is probably caused by FFT subpixels shifts
        # This avoids any filtering in the spatial / fourier domain

        BaseRendererModel.__init__(self, img_size, fit_params)
        
        xs = torch.linspace(-1, 1, self.img_size[0]*2)
        ys = torch.linspace(-1, 1, self.img_size[1]*2)
        xs, ys = torch.meshgrid(xs, ys, indexing='ij') # Guassian init
        gaussian = torch.exp(-(xs**2*32+ ys**2*32))
        self.template = ParameterModule(gaussian)
        self.template_act = nn.ReLU()
        
        self.template_pooling = nn.AvgPool2d((2,2), (2,2), padding=0)
        
        kx = torch.fft.fftshift(torch.fft.fftfreq(self.img_size[0]*2*2))
        ky = torch.fft.fftshift(torch.fft.fftfreq(self.img_size[1]*2*2))
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        self.register_buffer('KX', KX, False)
        self.register_buffer('KY', KY, False)   
    
    def _render_images(self, mapped_params, batch_size=None, ):
        # template is padded, shifted, croped and then down-sampled
        template = self._calculate_template()
        padding = (int(0.5*template.shape[0]),)*2 + (int(0.5*template.shape[1]),)*2
        template = nn.functional.pad(template.unsqueeze(0), padding, mode='replicate')[0]
        
        template_fft = torch.fft.fftshift(torch.fft.fft2(template))
        
        shifted_fft = template_fft.unsqueeze(0)*torch.exp(-2j*np.pi*(self.KX*mapped_params['x'] + self.KY*mapped_params['y']))
        
        shifted_template = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(shifted_fft, dim=(-2,-1))))
        
        shifted_template = shifted_template[:,:,
                                            padding[0]:-padding[1],
                                            padding[2]:-padding[3],
                                           ]
        shifted_template = self.template_pooling(shifted_template)
        
        shifted_template = shifted_template * mapped_params['A'] + mapped_params['bg']
        shifted_template = shifted_template * (mapped_params['p']>0.5)
        
        return shifted_template
    
    def _calculate_template(self, pool=False):
        template = self.template_act(self.template(None))
        if pool:
            template = self.template_pooling(template[None,None,...])[0,0]
        return template

    
class FourierOptics2DModel(BaseFitModel):
    def __init__(self, fit_params=['x', 'y'], *args, **kwargs):
        BaseFitModel.__init__(self, renderer_class=FourierOptics2DRenderer, fit_params=fit_params, *args, **kwargs)
    
    def get_suppl(self, colored=False):
        pupil_magnitude, pupil_phase, pupil_prop = self.renderer._calculate_pupil()
        pupil_magnitude = pupil_magnitude.detach().numpy()
        pupil_phase = pupil_phase.detach().numpy()
        pupil_prop = pupil_prop.detach().numpy()
        if colored:
            pupil_magnitude = util.color_images(pupil_magnitude, full_output=True)
            pupil_phase = util.color_images(pupil_phase, vsym=True, full_output=True)
            pupil_prop = util.color_images(pupil_prop, full_output=True)
        return {'pupil mag':pupil_magnitude, 'pupil phase':pupil_phase, 'z propagate':pupil_prop}

class FourierOptics2DRenderer(BaseRendererModel):
    def __init__(self, img_size, fit_params, pupil_params={'scale':0.75, 'apod':False}):

        BaseRendererModel.__init__(self, img_size, fit_params)
        
        xs = torch.linspace(-1., 1., self.img_size[0]) / pupil_params['scale']
        ys = torch.linspace(-1., 1., self.img_size[0]) / pupil_params['scale']
        XS, YS = torch.meshgrid(xs, ys, indexing='ij')
        R = torch.sqrt(XS**2 + YS**2)
        if pupil_params['apod']:
            pupil_magnitude = torch.sqrt(1-torch.minimum(R, torch.ones_like(R))**2)
        else:
            pupil_magnitude = (R <= 1).type(torch.float)
        if 'pupil' in fit_params:
            self.pupil_magnitude = nn.Sequential(
                ParameterModule(pupil_magnitude),
                nn.ReLU(),
                nn.Dropout(p=0.25),
            )
        else:
            self.pupil_magnitude = pupil_magnitude
            
        pupil_phase = torch.zeros((self.img_size[0], self.img_size[1]))
        nn.init.xavier_normal_(pupil_phase, 0.1)
        self.pupil_phase = nn.Sequential(
            ParameterModule(pupil_phase),
            nn.Identity(),
            nn.Dropout(p=0.25),
        )
        
        pupil_prop = torch.sqrt(1-torch.minimum(R, torch.ones_like(R))**2)
        if False:
            self.pupil_prop = nn.Sequential(
                ParameterModule(torch.ones(1)*0.1),
                nn.Identity(),
                nn.Dropout(p=0.25),
            )
        else:
            self.pupil_prop = pupil_prop            
            
        pupil_padding_factor = 4
        pupil_padding_clip = 0.5 * (pupil_padding_factor - 1)
        self.pupil_padding = (int(pupil_padding_clip*self.img_size[0]),)*2 + (int(pupil_padding_clip*self.img_size[1]),)*2
        
        kx = torch.fft.fftshift(torch.fft.fftfreq(self.img_size[0]*pupil_padding_factor))
        ky = torch.fft.fftshift(torch.fft.fftfreq(self.img_size[1]*pupil_padding_factor))
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        self.register_buffer('KX', KX, False)
        self.register_buffer('KY', KY, False)        
    
    def _render_images(self, mapped_params, batch_size, ):        
        pupil_magnitude, pupil_phase, pupil_prop = self._calculate_pupil()        
        pupil_phase = pupil_phase[None,...] * torch.ones([batch_size,]+[1,]*3)
        
        pupil_magnitude = nn.functional.pad(pupil_magnitude, self.pupil_padding, mode='constant')
        pupil_phase = nn.functional.pad(pupil_phase, self.pupil_padding, mode='constant')
        pupil_prop = nn.functional.pad(pupil_prop, self.pupil_padding, mode='constant')
        
        pupil_phase = pupil_phase - 2 * np.pi * (self.KX[None,...] * mapped_params['x'])
        pupil_phase = pupil_phase - 2 * np.pi * (self.KY[None,...] * mapped_params['y'])
        pupil_phase = pupil_phase + pupil_prop * mapped_params['z']
        
        pupil = pupil_magnitude[None,None,...] * torch.exp(1j * pupil_phase)

        shifted_pupil = torch.fft.ifftshift(pupil, dim=(-2,-1))
               
        images = torch.fft.ifftshift(torch.abs(torch.fft.ifft2(shifted_pupil))**2, dim=(-2,-1))
        images = images / torch.amax(images, dim=(2,3), keepdims=True)
        images = images[:,:,self.pupil_padding[0]:-self.pupil_padding[1],
                 self.pupil_padding[2]:-self.pupil_padding[3]]
        
        images = images * mapped_params['A'] + mapped_params['bg']
        images = images * (mapped_params['p']>0.5)
        
        return images
    
    def _calculate_pupil(self):
        if 'pupil' in self.fit_params:
            pupil_magnitude = self.pupil_magnitude(None)
        else:
            pupil_magnitude = self.pupil_magnitude
        pupil_phase = self.pupil_phase(None)
        pupil_prop = self.pupil_prop
        return pupil_magnitude, pupil_phase, pupil_prop

    
class ParameterModule(nn.Module):
    """
    Hack
    Wrapper class for nn.Parameter so that it will show up with pytorch.summary
    """
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        self.parameter = nn.Parameter(*args, **kwargs)        
    
    def forward(self, x=None):
        return self.parameter
    

def check_model(model, dataloader):
    features, labels = next(iter(dataloader))
    is_training = model.training
    model.train(False)
    pred = model(features)
    pred = pred.detach().numpy()
    
    print("input shape: {}, output_shape: {}".format(features.shape, pred.shape))
    
    fig, axes = plt.subplots(1, 2, figsize=(8,3))
    im = axes[0].imshow(features[0,0])
    plt.colorbar(im, ax=axes[0])
    axes[0].set_title("data")
    im = axes[1].imshow(pred[0,0])
    plt.colorbar(im, ax=axes[1])
    axes[1].set_title("predicted")
    
    if hasattr(model, 'get_suppl'):
        suppl_images = model.get_suppl(colored=True)
        fig, axes = plt.subplots(1, len(suppl_images), figsize=(4*len(suppl_images), 3), squeeze=False)
        for i, (key, (img, norm, cmap)) in enumerate(suppl_images.items()):
            im = axes[0, i].imshow(img)
            plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[0, i])
            axes[0, i].set_title(key)
        
    if hasattr(model, 'render_example_images'):
        
        example_images = model.render_example_images(8)
        fig, axes = plt.subplots(1, len(example_images), figsize=(4*len(example_images), 3), squeeze=False)
        for i, img in enumerate(example_images):
            im = axes[0, i].imshow(img[0])
            plt.colorbar(im, ax=axes[0, i])
            axes[0, i].set_title("E.g. {}".format(i))
            
        fig, axes = plt.subplots(1, len(example_images), figsize=(4*len(example_images), 3), squeeze=False)
        for i, img in enumerate(example_images):
            im = axes[0, i].imshow(np.log10(img[0]))
            plt.colorbar(im, ax=axes[0, i])
            axes[0, i].set_title("E.g. {}".format(i))            
            
    model.train(is_training)