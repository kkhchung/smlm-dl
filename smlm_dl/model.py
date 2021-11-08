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
        
    def render_images(self, params, detach=False):
        # explicit call to render images without going through the rest of the model
        raise NotImplementedError()
        
    def render_example_images(self, num):
        raise NotImplementedError()

        
class Gaussian2DModel(EncoderModel):
    def __init__(self, *args, **kwargs):
        EncoderModel.__init__(self, last_out_channels=3, *args, **kwargs)
        self.renderer = Gaussian2DRenderer(self.img_size)
        
    def forward(self, x):
        x = EncoderModel.forward(self, x)
        x = self.renderer(x)
        return x

    
class Gaussian2DRenderer(BaseRendererModel):
    def __init__(self, img_size):
        # currently fixed at x, y, sig
        BaseRendererModel.__init__(self, img_size,
                                   nn.ModuleList([nn.Tanh(),
                                                  nn.Tanh(),
                                                  nn.ReLU()]),
                                   [0,
                                    0,
                                    1,],
                                   [0.75 * img_size[0],
                                    0.75 * img_size[1],
                                    0.125 * 0.5 * sum(img_size),],
                                  )
    
    def render_images(self, params, as_numpy_arr=False):
        xs = torch.arange(0, self.img_size[0]) - 0.5*(self.img_size[0]-1)
        ys = torch.arange(0, self.img_size[1]) - 0.5*(self.img_size[1]-1)
        XS, YS = torch.meshgrid(xs, ys, indexing='ij')

        # images = torch.exp(-((XS[None,...]-params[:,[0]]*0.5*self.img_size[0])**2/(0.25*self.img_size[0]*(params[:,[2]]+1)) \
        #                + (YS[None,...]-params[:,[1]]*0.5*self.img_size[1])**2/(0.25*self.img_size[0]*(params[:,[2]]+1))))
        
        images = torch.exp(-((XS[None,...]-params[:,[0]])**2/params[:,[2]] \
                       + (YS[None,...]-params[:,[1]])**2/params[:,[2]]))
        
        images = images - torch.amin(images, dim=(2,3), keepdims=True)
        images = images / torch.amax(images, dim=(2,3), keepdims=True)
        
        # images = images * (x[:,[0]] + 1)
        # images = images + x[:,[1]]
        
        if as_numpy_arr:
            images = images.detach().numpy()
        
        return images
    
    def render_example_images(self, num):        
        params = torch.rand((num, self.n_params, 1, 1))
        params[:,0] = 2 * (params[:,0] - 0.5) * 0.75 * self.img_size[0]
        params[:,1] = 2 * (params[:,1] - 0.5) * 0.75 * self.img_size[1]
        params[:,2] = 4 * (params[:,2] + 1) * 0.125 * 0.5 * sum(self.img_size)
        images = self.render_images(params, True)
        return images
    
    
class Template2DModel(EncoderModel):
    def __init__(self, *args, **kwargs):
        EncoderModel.__init__(self, last_out_channels=2, *args, **kwargs)
        self.renderer = Template2DRenderer(self.img_size)
        
    def forward(self, x):
        x = EncoderModel.forward(self, x)
        x = self.renderer(x)
        return x
    
    def get_suppl(self):
        template = self.renderer.template.parameter.detach().numpy()
        return {'template':template}

    
class Template2DRenderer(BaseRendererModel):
    def __init__(self, img_size):
        BaseRendererModel.__init__(self, img_size,
                                   nn.ModuleList([nn.Tanh(),
                                                  nn.Tanh(),
                                                 ]),                                   
                                   [0,
                                    0,],
                                   [1 * img_size[0],
                                    1 * img_size[1],],
                          )
        
        xs = torch.linspace(-4, 4, self.img_size[0]*2)
        ys = torch.linspace(-4, 4, self.img_size[0]*2)
        xs, ys = torch.meshgrid(xs, ys, indexing='ij')
        # r = torch.sqrt(xs**2+ys**2)
        # self.template = ParameterModule(torch.clip(1-r, min=0))        
        
        r = torch.exp(-(xs**2/1 + ys**2/1))
        r = r / torch.amax(r)
        # r -= 0.5
        # r *= 2
        self.template = ParameterModule(r)
        self.template_render = nn.ReLU()
        
        # # tempalate image stored at 4 resolution
        # self.template = ParameterModule(torch.zeros((self.img_size[0]*2, self.img_size[1]*2)))
        # self.template.init_tensor()
        self.pooling = nn.AvgPool2d((2,2))
        
        kx = torch.fft.fftfreq(self.img_size[0]*2*2)
        ky = torch.fft.fftfreq(self.img_size[1]*2*2)
        self.kx, self.ky = torch.meshgrid(kx, ky, indexing='ij')
    
    def render_images(self, params, as_numpy_arr=False):
        template = self.template(None)
        template = self.template_render(template)
        
        template = nn.functional.pad(template.unsqueeze(0), (int(0.5*template.shape[0]),)*2 + (int(0.5*template.shape[1]),)*2, mode='replicate')[0]
        template_fft = torch.fft.fft2(template)
        # template_fft = torch.fft.fftshift(template)
        
        # shifted_fft = template_fft.unsqueeze(0)*torch.exp(-2j*np.pi*(self.kx*params[:,[0]]*0.75*self.img_size[0] + self.ky*params[:,[1]]*0.75*self.img_size[1]))
        shifted_fft = template_fft.unsqueeze(0)*torch.exp(-2j*np.pi*(self.kx*params[:,[0]] + self.ky*params[:,[1]]))

        shifted_template = torch.abs(torch.fft.ifft2(shifted_fft))
        # shifted_template = torch.fft.fftshift(shifted_template)
        
        shifted_template = shifted_template[:,:,
                                            int(0.25*template.shape[0]):-int(0.25*template.shape[0]),
                                            int(0.25*template.shape[1]):-int(0.25*template.shape[1]),
                                           ]
        # print(shifted_template.shape)
        shifted_template = self.pooling(shifted_template)
        
        if as_numpy_arr:
            shifted_template = shifted_template.detach().numpy()
        
        return shifted_template
    
    def render_example_images(self, num):        
        params = torch.rand((num, self.n_params, 1, 1))
        params[:,0] = 2 * (params[:,0] - 0.5) * 1 * self.img_size[0]
        params[:,1] = 2 * (params[:,1] - 0.5) * 1 * self.img_size[1]
        images = self.render_images(params, True)
        return images
    
class FourierOptics2DModel(EncoderModel):
    def __init__(self, *args, **kwargs):
        EncoderModel.__init__(self, last_out_channels=2, *args, **kwargs)
        self.renderer = FourierOptics2DRenderer(self.img_size)        
        
    def forward(self, x):
        x = EncoderModel.forward(self, x)
        x = self.renderer(x)
        return x
    
    def get_suppl(self):
        pupil = self.renderer.pupil.parameter.detach().numpy()
        return {'pupil mag':np.abs(pupil), 'pupil phase':np.angle(pupil)}

class FourierOptics2DRenderer(BaseRendererModel):
    def __init__(self, img_size):
        BaseRendererModel.__init__(self, img_size,
                                   nn.ModuleList([nn.Tanh(),
                                                  nn.Tanh(),
                                                 ]),                                   
                                   [0,
                                    0,],
                                   [0.75 * img_size[0],
                                    0.75 * img_size[1],],
                                  )
        
        kx = torch.fft.fftfreq(self.img_size[0]*4)
        ky = torch.fft.fftfreq(self.img_size[1]*4)
        self.kx, self.ky = torch.meshgrid(kx, ky, indexing='ij')
        
        xs = torch.linspace(-1.5, 1.5, self.img_size[0])
        ys = torch.linspace(-1.5, 1.5, self.img_size[0])
        xs, ys = torch.meshgrid(xs, ys, indexing='ij')
        r = torch.sqrt(xs**2+ys**2)
        self.pupil = ParameterModule((r<=1).type(torch.cfloat))
        # self.pupil = ParameterModule(torch.ones((self.img_size[0], self.img_size[1]), dtype=torch.cfloat))        
    
    def render_images(self, params, as_numpy_arr=False):
        pupil = self.pupil(None)
        pupil = nn.functional.pad(pupil, (int(1.5*pupil.shape[0]),)*2 + (int(1.5*pupil.shape[1]),)*2)
        
        pupil = torch.fft.fftshift(pupil)
        
        # shifted_pupil = pupil[None,...]*torch.exp(-2j*np.pi*(self.kx*params[:,[0]]*0.5*self.img_size[0] + self.ky*params[:,[1]]*0.5*self.img_size[1]))
        shifted_pupil = pupil[None,...]*torch.exp(-2j*np.pi*(self.kx*params[:,[0]] + self.ky*params[:,[1]]))
        
        images = torch.fft.fftshift(torch.abs(torch.fft.fft2(shifted_pupil))**2)
        images = images[:,:,int(0.375*images.shape[2]):-int(0.375*images.shape[3]),
                 int(0.375*images.shape[2]):-int(0.375*images.shape[3])]
        
        images = images - torch.amin(images, dim=(2,3), keepdims=True)
        images = images / torch.amax(images, dim=(2,3), keepdims=True)
        
        if as_numpy_arr:
            images = images.detach().numpy()
        
        return images
    
    def render_example_images(self, num):        
        params = torch.rand((num, self.n_params, 1, 1))
        params[:,0] = 2 * (params[:,0] - 0.5) * 0.75 * self.img_size[0]
        params[:,1] = 2 * (params[:,1] - 0.5) * 0.75 * self.img_size[1]
        images = self.render_images(params, True)
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