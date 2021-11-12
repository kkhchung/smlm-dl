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
    
class BaseRendererModel(nn.Module):
    """
    Base renderer class for a few standardized output functions
    """
    params_ref = dict()
    params_default = dict()
    params_ind = dict()
    
    def __init__(self, img_size, params_activation, params_offset, params_scaling):
        nn.Module.__init__(self)
        self.img_size = img_size
        self.n_params = len(params_activation)
        self.params_activation = params_activation
        self.params_offset = params_offset
        self.params_scaling = params_scaling
        
    def forward(self, x):
        x_new = torch.empty_like(x)
        for i, activation_func in enumerate(self.params_activation):
            x_new[:,i,...] = activation_func(x[:,i,...])
            x_new[:,i] = torch.add(x_new[:,i], self.params_offset[i])
            x_new[:,i] = torch.mul(x_new[:,i], self.params_scaling[i])
        return self.render_images(x_new)
    
    def build_params_config(self, img_size, fit_params, new_params_ref={}, new_params_default={}):        
        self.params_ref.update({
            'x': [nn.Tanh(), 0, 0.75 * img_size[0]],
            'y': [nn.Tanh(), 0, 0.75 * img_size[1]],
            'A': [nn.ReLU(), 0, 1000],
            'bg': [nn.Tanh(), 0, 500],
        })
        
        self.params_default.update({            
            'x': 0,
            'y': 0,
            'A': 1,
            'bg': 0,
        })
        
        self.params_ref.update(new_params_ref)
        self.params_default.update(new_params_default)        
                
        params_activation = list()
        params_offset = list()
        params_scale = list()
        
        for i, key in enumerate(fit_params):
            if not key in self.params_ref:
                raise Exception("fit_params not recognised")
            self.params_ind[key] = i
            ref = self.params_ref[key]
            params_activation.append(ref[0])
            params_offset.append(ref[1])
            params_scale.append(ref[2])
        
        return params_activation, params_offset, params_scale
    
    def map_params(self, params):
        mapped_params = dict()
        for key in self.params_ref:
            if key in self.params_ind:
                mapped_params[key] = params[:, [self.params_ind[key]]]
            else:
                mapped_params[key] = self.params_default[key]
        return mapped_params
        
    def render_images(self, params, detach=False):
        # explicit call to render images without going through the rest of the model
        raise NotImplementedError()
        
    def render_example_images(self, num):
        params = list()
        for i, (key, val) in enumerate(self.params_ind.items()):
            act, offset, scaling = self.params_ref[key]
            if isinstance(act, nn.Tanh):
                param = np.random.uniform(-1, 1, num)
            elif isinstance(act, nn.ReLU):
                param = np.random.uniform(0, 1, num)
            else:
                raise Exception("Unrecognized activation function. [{}]".format(type(act)))
            param = (param + offset) * scaling
            params.append(param)
        params = np.stack(params, axis=1)[...,None,None]        
        images = self.render_images(torch.from_numpy(params), True)
        return images

        
class Gaussian2DModel(EncoderModel):
    def __init__(self, fit_params=['x', 'y', 'sig'], *args, **kwargs):
        EncoderModel.__init__(self, last_out_channels=len(fit_params), *args, **kwargs)
        self.renderer = Gaussian2DRenderer(self.img_size, fit_params)
        
    def forward(self, x):
        x = EncoderModel.forward(self, x)
        x = self.renderer(x)
        return x

    
class Gaussian2DRenderer(BaseRendererModel):
    
    def __init__(self, img_size, fit_params):

        params_activation, params_offset, params_scale = self.build_params_config(img_size, fit_params)
        
        BaseRendererModel.__init__(self, img_size,
                                   nn.ModuleList(params_activation),
                                   params_offset,
                                   params_scale,
                                  )
        
        xs = torch.arange(0, self.img_size[0]) - 0.5*(self.img_size[0]-1)
        ys = torch.arange(0, self.img_size[1]) - 0.5*(self.img_size[1]-1)
        XS, YS = torch.meshgrid(xs, ys, indexing='ij')
        self.register_buffer('XS', XS, False)
        self.register_buffer('YS', YS, False)
                
    def build_params_config(self, img_size, fit_params):
        
        new_params_ref = {
            'sig': [nn.ReLU(), 2, 1],
            }
        
        new_params_default = {   
            'sig': 2,
        }
        
        return BaseRendererModel.build_params_config(self, img_size, fit_params, new_params_ref, new_params_default)
    
    def render_images(self, params, as_numpy_arr=False):
        mapped_params = self.map_params(params)
        
        images = torch.exp(-((self.XS[None,...]-mapped_params['x'])**2/mapped_params['sig'] \
                       + (self.YS[None,...]-mapped_params['y'])**2/mapped_params['sig']))
        
        images = images * mapped_params['A'] + mapped_params['bg']
        
        if as_numpy_arr:
            images = images.detach().numpy()
        
        return images
    
    
class Template2DModel(EncoderModel):
    def __init__(self, fit_params=['x', 'y'], *args, **kwargs):
        EncoderModel.__init__(self, last_out_channels=len(fit_params), *args, **kwargs)
        self.renderer = Template2DRenderer(self.img_size, fit_params)
        
    def forward(self, x):
        x = EncoderModel.forward(self, x)
        x = self.renderer(x)
        return x
    
    def get_suppl(self):
        template = self.renderer.template.parameter.detach().numpy() * self.renderer.hann_window.detach().numpy()
        template /= template.max()
        return {'template':template}

    
class Template2DRenderer(BaseRendererModel):
    def __init__(self, img_size, fit_params):
        params_activation, params_offset, params_scale = BaseRendererModel.build_params_config(self, img_size, fit_params)
        
        BaseRendererModel.__init__(self, img_size,
                                   nn.ModuleList(params_activation),                                   
                                   params_offset,
                                   params_scale,
                          )
        
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
    
    def render_images(self, params, as_numpy_arr=False):
        mapped_params = self.map_params(params)
        
        template = self.template(None)
        template = self.template_render(template)
        template = template * self.hann_window
        template = template / torch.amax(template)
        
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
        
        if as_numpy_arr:
            shifted_template = shifted_template.detach().numpy()
        
        return shifted_template

    
class FourierOptics2DModel(EncoderModel):
    def __init__(self, fit_params=['x', 'y'], *args, **kwargs):
        EncoderModel.__init__(self, last_out_channels=len(fit_params), *args, **kwargs)
        self.renderer = FourierOptics2DRenderer(self.img_size, fit_params)        
        
    def forward(self, x):
        x = EncoderModel.forward(self, x)
        x = self.renderer(x)
        return x
    
    def get_suppl(self):
        pupil_magnitude = self.renderer.pupil_magnitude.parameter.detach().numpy() * self.renderer.hann_window.detach().numpy()
        pupil_phase = self.renderer.pupil_phase.parameter.detach().numpy()
        return {'pupil mag':pupil_magnitude, 'pupil phase':pupil_phase}

class FourierOptics2DRenderer(BaseRendererModel):
    def __init__(self, img_size, fit_params):
        params_activation, params_offset, params_scale = self.build_params_config(img_size, fit_params)
        
        BaseRendererModel.__init__(self, img_size,
                                   nn.ModuleList(params_activation),
                                   params_offset,
                                   params_scale,
                                  )
        
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
        self.pupil_phase = ParameterModule(torch.zeros((self.img_size[0], self.img_size[1])))
        self.pupil_phase_act = nn.Tanh()
    
    def render_images(self, params, as_numpy_arr=False):
        mapped_params = self.map_params(params)
        
        pupil_magnitude = self.pupil_magnitude_act(self.pupil_magnitude(None))
        pupil_magnitude = pupil_magnitude * self.hann_window
        
        pupil_phase = self.pupil_phase_act(self.pupil_phase(None)) * np.pi
        pupil_phase = pupil_phase[None,...] * torch.ones(params.shape[0])[:,None,None,None]
        
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
        
        if as_numpy_arr:
            images = images.detach().numpy()
        
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
        
    if hasattr(model, 'renderer') and hasattr(model.renderer, 'render_example_images'):
        example_images = model.renderer.render_example_images(8)
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