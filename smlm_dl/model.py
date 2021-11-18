import numpy as np

import torch
from torch import nn

# from torchinfo import summary

from matplotlib import pyplot as plt

import warnings

class EncoderModel(nn.Module):
    def __init__(self, img_size=(32,32), depth=3, first_layer_out_channels=16, last_out_channels=2):
        nn.Module.__init__(self)
        if not(depth == 3) or not(img_size==(32,32)):
            warnings.warn("Not actually tested with other images sizes or model depth. On TODO list.")
            
        self.depth = depth
        self.img_size = img_size
        self.test = nn.Parameter(torch.ones(32, 32))
        
        self.encoders = nn.ModuleDict()
        for i in range(self.depth):
            in_channels = 1 if i == 0 else 2**(i-1) * first_layer_out_channels
            out_channels = 2**i * first_layer_out_channels

            self.encoders["conv_layer{}".format(i)] = self._conv_block(in_channels, out_channels)
            self.encoders["pool_layer{}".format(i)] = nn.MaxPool2d(2)
            self.encoders["dropout_layer{}".format(i)] = nn.Dropout2d(0.5)
            
        self.neck = nn.ModuleDict()
        self.neck["conv"] = self._conv_block(int(2**(self.depth-1) * first_layer_out_channels),
                                             (2**(self.depth) * first_layer_out_channels))
        self.neck["dropout"] = nn.Dropout2d(0.5)
        
        self.decoders = nn.ModuleDict()
        self.decoders["conv_layer"] = nn.Sequential(            
            nn.Conv2d(2**(self.depth) * first_layer_out_channels, last_out_channels, kernel_size=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(last_out_channels, last_out_channels, kernel_size=3, padding=0),
            # nn.Tanh(), # need to define last activation in the full models
            )
        
    def forward(self, x):
        for i, (key, module) in enumerate(self.encoders.items()):
            x = module(x)
            
        for i, (key, module) in enumerate(self.neck.items()):
            x = module(x)
            
        for i, (key, module) in enumerate(self.decoders.items()):
            x = module(x)
            
        return x
    
    def _conv_block(self, in_channels, out_channels):    
        return nn.Sequential(
            nn.GroupNorm(in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
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
    
    def __init__(self, img_size):
        nn.Module.__init__(self)
        self.img_size = img_size
        
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
        
        BaseRendererModel.__init__(self, img_size,)
        
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
    
    def get_suppl(self):
        template = self.renderer.template.parameter.detach().numpy() * self.renderer.hann_window.detach().numpy()
        template /= template.max()
        return {'template':template}

    
class Template2DRenderer(BaseRendererModel):
    def __init__(self, img_size, fit_params):

        BaseRendererModel.__init__(self, img_size,)
        
        xs = torch.linspace(-4, 4, self.img_size[0]*2)
        ys = torch.linspace(-4, 4, self.img_size[0]*2)
        xs, ys = torch.meshgrid(xs, ys, indexing='ij')
        r = torch.sqrt(xs**2+ys**2)
        # self.template = ParameterModule(torch.clip(1-r, min=0))
        hann_window = torch.cos(r/4)**2 * (r<=4)        
        self.register_buffer('hann_window', hann_window, False)
        
        xs = torch.linspace(-1, 1, self.img_size[0]*2*2)
        ys = torch.linspace(-1, 1, self.img_size[0]*2*2)
        xs, ys = torch.meshgrid(xs, ys, indexing='ij')
        r = torch.sqrt(xs**2+ys**2)
        large_hann_window = torch.cos(r/1)**2 * (r<=1)
        self.register_buffer('hann_window_large', large_hann_window, False)
        
        xs = torch.linspace(-4, 4, self.img_size[0]*2)
        ys = torch.linspace(-4, 4, self.img_size[0]*2)
        xs, ys = torch.meshgrid(xs, ys, indexing='ij')
        r = torch.exp(-(xs**2/1 + ys**2/1))
        # r = r / torch.amax(r)
        # r -= 0.5
        # r *= 2
        self.template = ParameterModule(r)
        self.template_render = nn.ReLU()
        
        # # tempalate image stored at 4 resolution
        # self.template = ParameterModule(torch.zeros((self.img_size[0]*2, self.img_size[1]*2)))
        # self.template.init_tensor()
        self.pooling = nn.AvgPool2d((3,3), (2,2), padding=(1,1))
        
        kx = torch.fft.fftfreq(self.img_size[0]*2*2)
        ky = torch.fft.fftfreq(self.img_size[1]*2*2)
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        self.register_buffer('KX', KX, False)
        self.register_buffer('KY', KY, False)
    
    def _render_images(self, mapped_params, batch_size=None, ):
        
        template = self.template(None)
        template = self.template_render(template)
        template = template * self.hann_window
        
        template = nn.functional.pad(template.unsqueeze(0), (int(0.5*template.shape[0]),)*2 + (int(0.5*template.shape[1]),)*2, mode='replicate')[0]
        template_fft = torch.fft.fft2(template)
        # template_fft = torch.fft.fftshift(template)
        
        # shifted_fft = template_fft.unsqueeze(0)*torch.exp(-2j*np.pi*(self.kx*params[:,[0]]*0.75*self.img_size[0] + self.ky*params[:,[1]]*0.75*self.img_size[1]))
        shifted_fft = template_fft.unsqueeze(0)*torch.exp(-2j*np.pi*(self.KX*mapped_params['x'] + self.KY*mapped_params['y']))

        shifted_fft = shifted_fft * torch.fft.fftshift(self.hann_window_large)
        
        shifted_template = torch.abs(torch.fft.ifft2(shifted_fft))
        # shifted_template = torch.fft.fftshift(shifted_template)
        
        
        shifted_template = shifted_template[:,:,
                                            int(0.25*template.shape[0]):-int(0.25*template.shape[0]),
                                            int(0.25*template.shape[1]):-int(0.25*template.shape[1]),
                                           ]
        # print(shifted_template.shape)
        shifted_template = self.pooling(shifted_template)
        
        shifted_template = shifted_template * mapped_params['A'] + mapped_params['bg']
        shifted_template = shifted_template * (mapped_params['p']>0.5)
        
        return shifted_template

    
class FourierOptics2DModel(BaseFitModel):
    def __init__(self, fit_params=['x', 'y'], *args, **kwargs):
        BaseFitModel.__init__(self, renderer_class=FourierOptics2DRenderer, fit_params=fit_params, *args, **kwargs)
    
    def get_suppl(self):
        pupil_magnitude = self.renderer.pupil_magnitude.parameter.detach().numpy() * self.renderer.hann_window.detach().numpy()
        pupil_phase = self.renderer.pupil_phase.parameter.detach().numpy()
        return {'pupil mag':pupil_magnitude, 'pupil phase':pupil_phase}

class FourierOptics2DRenderer(BaseRendererModel):
    def __init__(self, img_size, fit_params):

        BaseRendererModel.__init__(self, img_size,)
        
        kx = torch.fft.fftshift(torch.fft.fftfreq(self.img_size[0]))
        ky = torch.fft.fftshift(torch.fft.fftfreq(self.img_size[1]))
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        self.register_buffer('KX', KX, False)
        self.register_buffer('KY', KY, False)
        
        xs = torch.linspace(-1., 1., self.img_size[0])
        ys = torch.linspace(-1., 1., self.img_size[0])
        xs, ys = torch.meshgrid(xs, ys, indexing='ij')
        r = torch.sqrt(xs**2+ys**2)
        hann_window = torch.cos(r)**2 * (r<=1)
        # self.hann_window = hann_window
        self.register_buffer('hann_window', hann_window, False)
        
        self.pupil_magnitude = ParameterModule(torch.ones((self.img_size[0], self.img_size[1])))
        self.pupil_magnitude_act = nn.ReLU()
        
        pupil_phase_init = torch.zeros((self.img_size[0], self.img_size[1]))
        nn.init.xavier_normal_(pupil_phase_init, 0.1)
        self.pupil_phase = ParameterModule(pupil_phase_init)
        self.pupil_phase_act = nn.Tanh()
    
    def _render_images(self, mapped_params, batch_size, ):
        
        pupil_magnitude = self.pupil_magnitude_act(self.pupil_magnitude(None))
        pupil_magnitude = pupil_magnitude * self.hann_window
        
        pupil_phase = self.pupil_phase_act(self.pupil_phase(None)) * np.pi
        pupil_phase = pupil_phase[None,...] * torch.ones(batch_size)[:,None,None,None]
        
        pupil_phase = pupil_phase - np.pi * (self.KX[None,...] * mapped_params['x'])
        pupil_phase = pupil_phase - np.pi * (self.KY[None,...] * mapped_params['y'])
        
        pupil = pupil_magnitude[None,None,...] * torch.exp(1j * pupil_phase)
        pupil = nn.functional.pad(pupil, (int(1.5*pupil.shape[2]),)*2 + (int(1.5*pupil.shape[3]),)*2, mode='constant')

        shifted_pupil = torch.fft.ifftshift(pupil)
               
        images = torch.fft.ifftshift(torch.abs(torch.fft.ifft2(shifted_pupil))**2)
        images = images / torch.amax(images, dim=(2,3), keepdims=True)
        images = images[:,:,int(0.375*images.shape[2]):-int(0.375*images.shape[2]),
                 int(0.375*images.shape[3]):-int(0.375*images.shape[3])]
        
        images = images * mapped_params['A'] + mapped_params['bg']
        images = images * (mapped_params['p']>0.5)
        
        return images

    
class ParameterModule(nn.Module):
    """
    Hack
    Wrapper class for nn.Parameter so that it will show up with pytorch.summary
    """
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        self.parameter = nn.Parameter(*args, **kwargs)        
    
    def forward(self, x):
        return self.parameter
    
    def init_tensor(self):
        nn.init.xavier_normal_(self.parameter)
        shape = self.parameter.shape
        self.parameter.data[int(shape[0]*0.375):-int(shape[0]*0.375),
                            int(shape[1]*0.375):-int(shape[1]*0.375),
                           ] += 0.5
    

def check_model(model, dataloader):
    features, labels = next(iter(dataloader))
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
        suppl_images = model.get_suppl()
        fig, axes = plt.subplots(1, len(suppl_images), figsize=(4*len(suppl_images), 3), squeeze=False)
        for i, (key, img) in enumerate(suppl_images.items()):
            im = axes[0, i].imshow(img)
            plt.colorbar(im, ax=axes[0, i])
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