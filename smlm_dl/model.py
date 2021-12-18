import functools
import numpy as np

import torch
from torch import nn

import matplotlib
from matplotlib import pyplot as plt
from skimage import restoration

import warnings

import util, zernike

class BaseModel(nn.Module):
    def call_auto(self, x, y=None):
        # 'smart' call that workers for both id / image type encoder
        if y is None:
            return nn.Module.__call__(self, x)
        else:
            if self.image_input:
                pred = nn.Module.__call__(self, x)
            else:
                pred = nn.Module.__call__(self, y["id"])
            return pred

class BaseEncoderModel(BaseModel):
    def __init__(self, last_out_channels=2, **kwargs):
        BaseModel.__init__(self)
        self.last_out_channels = last_out_channels
        self.build_model(**kwargs)
        
    def build_model(self, **kwargs):
        raise NotImplementedError()


class IdEncoderModel(BaseEncoderModel):
    def __init__(self, num_img, last_out_channels=2, init_weights=None):
        BaseEncoderModel.__init__(self, last_out_channels=last_out_channels, **{"num_img":num_img, "init_weights":init_weights})
        self.image_input = False
    
    def build_model(self, num_img, init_weights=None):
        self.one_hot = functools.partial(torch.nn.functional.one_hot, num_classes=num_img)
        self.encoders = nn.ModuleDict()
        self.encoders["scale"] = nn.Linear(num_img, self.last_out_channels, bias=False)
        if not init_weights is None:
            with torch.no_grad():
                print(init_weights.shape)
                print(self.encoders["scale"].weight.shape)
                self.encoders["scale"].weight.copy_(torch.as_tensor(init_weights))
    
    def forward(self, x):
        x = x.to(torch.long) 
        x = self.one_hot(x)
        x = x.to(torch.float)
        for key, val in self.encoders.items():
            x = val(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        return x
    
    
class ImageEncoderModel(BaseEncoderModel):
    def __init__(self, img_size=(32,32), in_channels=1, last_out_channels=2, **kwargs):
        if not img_size is None and (img_size[0] % 2 != 0 or img_size[1] % 2 !=0):
            raise Exception("Image input size needs to be multiples of two.")
        self.image_input = True
        self.img_size = img_size
        self.in_channels = in_channels
        BaseEncoderModel.__init__(self, last_out_channels=last_out_channels, **kwargs)
        

class ConvImageEncoderModel(ImageEncoderModel):
    def __init__(self, img_size=(32,32), depth=3, in_channels=1, first_layer_out_channels=16, last_out_channels=2, skip_channels=0,):
        if 2**depth > img_size[0] or 2**depth > img_size[1]:
            raise Exception("Model too deep for this image size (depth = {}, image size needs to be at least ({}, {}))".format(depth, 2**depth, 2**depth))
        
        ImageEncoderModel.__init__(self, img_size=img_size, in_channels=in_channels,
                                  last_out_channels=last_out_channels,
                                  **{"depth":depth, "first_layer_out_channels":first_layer_out_channels, "skip_channels":skip_channels})
        
    def build_model(self, depth, first_layer_out_channels, skip_channels):        
        self.encoders = nn.ModuleDict()
        self.skips = nn.ModuleDict()
        for i in range(depth):
            in_channels = self.in_channels if i == 0 else 2**(i-1) * first_layer_out_channels
            out_channels = 2**i * first_layer_out_channels            
            if skip_channels > 0:
                in_shape = (int(self.img_size[0] * 0.5**i), int(self.img_size[1] * 0.5**i))
                self.skips["skip_conv_layer{}".format(i)] = self._skip_block(in_channels, skip_channels, in_shape)
            self.encoders["conv_layer{}".format(i)] = self._encode_block(in_channels, out_channels)
            
        self.neck = nn.ModuleDict()
        self.neck["conv_layer_0"] = self._neck_block(int(2**(depth-1) * first_layer_out_channels),
                                             (2**(depth) * first_layer_out_channels))
        self.neck["conv_layer_1"] = self._encode_final_block(2**(depth) * first_layer_out_channels,
                                                             2**(depth-1) * first_layer_out_channels,
                                                            tuple(int(size*0.5**depth) for size in self.img_size))
        
        self.decoders = nn.ModuleDict()
        self.decoders["dense_layer_0"] = nn.Conv2d(2**(depth-1) * first_layer_out_channels + skip_channels*depth,
                                                   self.last_out_channels, kernel_size=1, padding=0)
        
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
            # nn.GroupNorm(in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.GroupNorm(out_channels, out_channels),
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
    
    
class FeedbackModel(BaseModel):
    def __init__(self, img_size=(32,32), feedback_size=(32,32)):
        BaseModel.__init__(self)
        
    def forward(self, x, feedback):
        raise NotImplementedError()
        
    
class DirectConcatFeedbackModel(FeedbackModel):
    def __init__(self, img_size=(32,32), feedback_size=(32,32)):
        if not all([img_size[d]==feedback_size[d] for d in range(len(img_size))]):
            raise Exception("Input and feedback need to have identical H and W.")
        FeedbackModel.__init__(self)
        self.norm = nn.GroupNorm(1, 1)
        
    def forward(self, x, feedback):
        feedback = torch.tile(feedback.detach(), (x.shape[0], 1, 1, 1))
        x = torch.cat([self.norm(x), feedback], dim=1)
        return x
    
class CropAndConcatFeedbackModel(FeedbackModel):
    def __init__(self, img_size=(32,32), feedback_size=(32,32)):
        FeedbackModel.__init__(self)
        self.norm = nn.GroupNorm(1, 1)
        self.padding = [int((feedback_size[d]-img_size[d])/2) for d in range(len(feedback_size))]
        
    def forward(self, x, feedback):
        feedback = torch.tile(feedback[:,:,self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1]], (x.shape[0], 1, 1, 1))
        x = torch.cat([self.norm(x), feedback], dim=1)
        return x
    

class DenseFeedbackModel(FeedbackModel):
    def __init__(self, img_size=(32,32), feedback_size=(32,32), feedback_features=3, depth=1):
        FeedbackModel.__init__(self)
        self.norm = nn.GroupNorm(1, 1)
        self.feedback = nn.ModuleDict()
        for i in range(depth):
            self.feedback['dense_layer_{}'.format(i)] = self.dense_block(feedback_size[-1], feedback_size[-1], feedback_features)
        
    def forward(self, x, feedback):
        feedback = feedback.detach()
        for key, val in self.feedback.items():
            feedback = val(feedback)
        feedback = torch.tile(feedback, (x.shape[0], 1, 1, 1))
        x = torch.cat([self.norm(x), feedback], dim=1)
        return x
            
    def dense_block(self, in_channels, out_channels, features):
        return nn.Sequential(
            nn.GroupNorm(1, features),
            nn.Linear(in_channels, out_channels, ),
            nn.ReLU(),
            nn.GroupNorm(1, features),
            nn.Linear(out_channels, out_channels, ),
            nn.ReLU(),
            nn.Dropout2d(0.5),
        )

    
class DiffFeedbackModel(FeedbackModel):
    def __init__(self, img_size=(32,32), feedback_size=(32,32)):
        FeedbackModel.__init__(self)        
        
    def forward(self, x, feedback):
        feedback = feedback.detach()
        # feedback = self.normalize(feedback.detach())
        # feedback = torch.tile(feedback, (x.shape[0], 1, 1, 1))
        # feedback = self.scale_to(feedback, x)
        x = torch.cat([x, feedback-x], dim=1)
        return x
    
    def normalize(self, arr):
        arr -= arr.min()
        arr /= arr.max()
        return arr
            
    def scale_to(self, feedback, x):
        feedback += torch.amin(x, dim=(-2,-1), keepdim=True)
        feedback *= torch.amax(x, dim=(-2,-1), keepdim=True) - torch.amin(x, dim=(-2,-1), keepdim=True)
        return feedback
    
    
class BaseFitModel(BaseModel):
    
    def __init__(self, renderer_class, encoder_class, feedback_class=None,
                 img_size=(32,32), fit_params=['x', 'y', ], max_psf_count=1,
                 params_ref_override={}, params_ref_no_scale=False,
                 encoder_params={}, renderer_params={}, feedback_params={},):
        BaseModel.__init__(self)
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
        if issubclass(encoder_class, ImageEncoderModel):
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
    
    
class BaseRendererModel(BaseModel):
    """
    Base renderer class for a few standardized output functions
    """
    params_ref = dict()
    params_default = dict()
    params_ind = dict()
    
    def __init__(self, img_size, fit_params):
        BaseModel.__init__(self)
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
    
    def get_feedback(self):
        raise NotImplementedError()

        
class Gaussian2DModel(BaseFitModel):
    def __init__(self, fit_params=['x', 'y', 'sig'], encoder_class=ConvImageEncoderModel, **kwargs):
        BaseFitModel.__init__(self, encoder_class=encoder_class, renderer_class=Gaussian2DRenderer, fit_params=fit_params, **kwargs)
    
    def setup_fit_params(self, img_size, fit_params, max_psf_count, new_params_ref, params_ref_no_scale):
        params_ref = {
            'sig': FitParameter(nn.ReLU(), 2, 1, 5, True),
        }
        params_ref.update(new_params_ref)
        
        BaseFitModel.setup_fit_params(self, img_size, fit_params, max_psf_count, params_ref, params_ref_no_scale)

    
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
    def __init__(self, fit_params=['x', 'y'], encoder_class=ConvImageEncoderModel, **kwargs):
        BaseFitModel.__init__(self, encoder_class=encoder_class, renderer_class=Template2DRenderer, fit_params=fit_params,
                              **kwargs)
    
    def get_suppl(self, colored=False):
        res = {'images': {},
               }
        template = self.renderer._calculate_template(True).detach().numpy()
        template_2x = self.renderer._calculate_template(False).detach().numpy()
        if colored:
            template = util.color_images(template, full_output=True)
            template_2x = util.color_images(template_2x, full_output=True)
        res['images'].update({'template':template, 'template 2x':template_2x, })
        
        if not self.renderer.conv is None:
            conv_kernel = self.renderer.conv(None).detach().numpy()[0,0]
            conv_kernel = util.color_images(conv_kernel, full_output=True)
            res['images']['conv'] = conv_kernel
        return res

    
class Template2DRenderer(BaseRendererModel):
    def __init__(self, img_size, fit_params, template_init=None, template_padding=None, conv=None):
        # The 2x sampling avoids the banding artifact that is probably caused by FFT subpixels shifts
        # This avoids any filtering in the spatial / fourier domain

        BaseRendererModel.__init__(self, img_size, fit_params)
        
        scale_factor = 2
        self.register_buffer('scale_factor', torch.tensor(scale_factor))
        if template_padding is None:
            # template_padding = [int((s+3)/4)*2 for s in img_size]
            template_padding = [0, 0]
        template_padding_scaled = [s*scale_factor for s in template_padding]
        self.register_buffer('template_padding_scaled', torch.tensor(template_padding_scaled), False)
        template_size_scaled = [img_size[d]*scale_factor + self.template_padding_scaled[d]*2 for d in range(2)]
        noise = torch.zeros(template_size_scaled)
        if template_init is None or template_init == 'gauss':
            nn.init.xavier_uniform_(noise, 0.1)
            noise += 0.1 + 1e-3
             # Guassian init
            xs = torch.linspace(-1, 1, template_size_scaled[0])
            ys = torch.linspace(-1, 1, template_size_scaled[1])
            xs, ys = torch.meshgrid(xs, ys, indexing='ij')
            template = torch.exp(-(xs**2*32+ ys**2*32))
        else:
            nn.init.xavier_uniform_(noise, 0.5)
            noise += 0.5 + 1e-3
            template = torch.tensor(template_init)
            template = template - np.percentile(template, 10)
            template = template / template.max()
            template = nn.functional.interpolate(template[None,None,...], scale_factor=scale_factor)
            template = nn.functional.pad(template, (self.template_padding_scaled[0],)*2 + (self.template_padding_scaled[1],)*2, mode='reflect', )# value=template.mean())
            template = template[0,0]
        self.template = nn.Sequential(ParameterModule(template + noise),
                                      nn.ReLU(),
                                      # nn.Dropout2d(p=0.25),
                                     )
        
        self.template_pooling = nn.AvgPool2d((scale_factor,scale_factor), (scale_factor,scale_factor), padding=0)
        
        kx = torch.fft.fftshift(torch.fft.fftfreq(template_size_scaled[0]*2))
        ky = torch.fft.fftshift(torch.fft.fftfreq(template_size_scaled[1]*2))
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        self.register_buffer('KX', KX, False)
        self.register_buffer('KY', KY, False)
        
        if not conv is None:            
            self.conv = nn.Sequential(ParameterModule(torch.ones(1, 1, conv, conv) / conv**2),
                                      nn.ReLU(),
                                     )
        else:
            self.conv = None
    
    def _render_images(self, mapped_params, batch_size=None, ):
        # template is padded, shifted, croped and then down-sampled
        template = self._calculate_template()
        padding = (int(0.5*template.shape[0]),)*2 + (int(0.5*template.shape[1]),)*2
        template = nn.functional.pad(template.unsqueeze(0).unsqueeze(0), padding, mode='constant')[0,0]
        
        template_fft = torch.fft.fftshift(torch.fft.fft2(template))
        
        shifted_fft = template_fft.unsqueeze(0)*torch.exp(-2j*np.pi*(self.KX*mapped_params['x'] + self.KY*mapped_params['y'])*self.scale_factor)
        
        shifted_template = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(shifted_fft, dim=(-2,-1))))
        
        shifted_template = shifted_template[:,:,
                                            padding[0]+self.template_padding_scaled[0]:-padding[1]-self.template_padding_scaled[0],
                                            padding[2]+self.template_padding_scaled[1]:-padding[3]-self.template_padding_scaled[1],
                                           ]
        shifted_template = self.template_pooling(shifted_template)
        
        if not self.conv is None:
            kernel = self.conv(None)
            shifted_template = torch.nn.functional.pad(shifted_template, (kernel.shape[-1]//2,)*2 + (kernel.shape[-2]//2,)*2, mode="reflect")
            shifted_template = torch.nn.functional.conv2d(shifted_template, kernel, padding=0)
        
        shifted_template = shifted_template * mapped_params['A'] + mapped_params['bg']
        shifted_template = shifted_template * (mapped_params['p']>0.5)
        
        return shifted_template
    
    def _calculate_template(self, pool=False):
        template = self.template(None)
        if pool:
            template = self.template_pooling(template[None,None,...])[0,0]
        return template
    
    def get_feedback(self):
        return self._calculate_template(True).unsqueeze(0).unsqueeze(0)

    
class FourierOptics2DModel(BaseFitModel):
    def __init__(self, fit_params=['x', 'y'], encoder_class=ConvImageEncoderModel, **kwargs):
        BaseFitModel.__init__(self, encoder_class=encoder_class, renderer_class=FourierOptics2DRenderer, fit_params=fit_params, **kwargs)
    
    def get_suppl(self, colored=False):
        pupil_magnitude, pupil_phase, pupil_prop = self.renderer._calculate_pupil()
        pupil_magnitude = pupil_magnitude.detach().numpy()
        pupil_phase = pupil_phase.detach().numpy()
        pupil_prop = pupil_prop.detach().numpy()
        
        zern_coeffs = zernike.fit_zernike_from_pupil(pupil_magnitude*np.exp(1j*pupil_phase), 16, self.renderer.R, np.arctan2(self.renderer.US, self.renderer.VS))
        zern_plot = functools.partial(zernike.plot_zernike_coeffs, zernike_coeffs=zern_coeffs)
        
        if colored:
            pupil_magnitude = util.color_images(pupil_magnitude, full_output=True)
            pupil_phase = util.color_images(pupil_phase, vsym=True, full_output=True)
            pupil_prop = util.color_images(pupil_prop, full_output=True)
            
        return {'images':{'pupil mag':pupil_magnitude, 'pupil phase':pupil_phase, 'z propagate':pupil_prop},
                'plots':{'zernike': {'plot':zern_plot, 'kwargs':{'figsize':(8,3)}}},
               }

class FourierOptics2DRenderer(BaseRendererModel):
    def __init__(self, img_size, fit_params, pupil_params={'scale':0.75, 'apod':False, 'phase_init_zern':{}},):

        BaseRendererModel.__init__(self, img_size, fit_params)
        
        us = torch.linspace(-1., 1., self.img_size[0]) / pupil_params['scale']
        vs = torch.linspace(-1., 1., self.img_size[0]) / pupil_params['scale']
        US, VS = torch.meshgrid(us, vs, indexing='ij')
        R = torch.sqrt(US**2 + VS**2)
        self.register_buffer('US', US, False)
        self.register_buffer('VS', VS, False)
        self.register_buffer('R', R, False)
        
        if pupil_params['apod']:
            pupil_magnitude = torch.sqrt(1-torch.minimum(R, torch.ones_like(R))**2)
        else:
            pupil_magnitude = (R <= 1).type(torch.float)
        if 'pupil' in fit_params:
            self.pupil_magnitude = nn.Sequential(
                ParameterModule(pupil_magnitude),
                nn.ReLU(),
                # nn.Dropout(p=0.25),
            )
        else:
            self.pupil_magnitude = pupil_magnitude
            
        pupil_phase = torch.zeros((self.img_size[0], self.img_size[1]))
        nn.init.xavier_normal_(pupil_phase, 0.1)
        pupil_phase = pupil_phase + zernike.calculate_pupil_phase(R, torch.atan2(US, VS), pupil_params.get("phase_init_zern", {}))
        self.pupil_phase = nn.Sequential(
            ParameterModule(pupil_phase),
            nn.Identity(),
            # nn.Dropout(p=0.25),
        )
        
        azimuthal_angle = torch.atan2(US, VS)
        self.register_buffer('mask', R <=1)
        self.register_buffer('zern_tilt', zernike.calculate_pupil_phase(self.R, azimuthal_angle, {1:1}))
        self.register_buffer('zern_tip', zernike.calculate_pupil_phase(self.R, azimuthal_angle, {2:1}))
        self.register_buffer('zern_defocus', zernike.calculate_pupil_phase(self.R, azimuthal_angle, {4:1}))
        
        pupil_prop = torch.sqrt(1-torch.minimum(R, torch.ones_like(R))**2)
        if False:
            self.pupil_prop = nn.Sequential(
                ParameterModule(torch.ones(1)*0.1),
                nn.Identity(),
                # nn.Dropout(p=0.25),
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
        self._substract_tilt_tip_defocus(pupil_phase)
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
    
    def _substract_tilt_tip_defocus(self, pupil_phase):
        pupil_phase_masked = restoration.unwrap_phase(np.ma.array(pupil_phase.detach().numpy(), mask=~self.mask))
        pupil_phase_masked = torch.as_tensor(pupil_phase_masked.data[self.mask], dtype=torch.float)
        with torch.no_grad():
            for basis in [self.zern_defocus, self.zern_tip, self.zern_tilt]:
                res = torch.linalg.lstsq(basis[self.mask].reshape(-1, 1), pupil_phase_masked.reshape(-1, 1))[0]
                pred = basis * res
                pupil_phase -= pred * self.mask
    
    def get_feedback(self):
        return torch.stack(self._calculate_pupil()).unsqueeze(0)

    
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
    

def check_model(model, dataloader=None):
    is_training = model.training
    model.train(False)
    
    if not dataloader is None:
        features, labels = next(iter(dataloader))
        pred = model.call_auto(features, labels)
        pred = pred.detach().numpy()
        features = features.detach().numpy()
    
        print("input shape: {}, output_shape: {}".format(features.shape, pred.shape))
        
        pred_is_image = pred.shape[2]>1 and pred.shape[3]>1
        
        n_col = 8
        n_images = 8 
        features_tiled, n_col, n_row = util.tile_images(features[:n_images].mean(1), n_col)
        
        fig, axes = plt.subplots(3 if pred_is_image else 1, 1,
                                 figsize=(4*n_col, 3*n_row*(3 if pred_is_image else 1)),
                                 squeeze=False)  
        im = axes[0,0].imshow(features_tiled)
        plt.colorbar(im, ax=axes[0,0])
        axes[0,0].set_title("data")
        
        if pred_is_image:
            pred_tiled, n_col, n_row = util.tile_images(pred[:n_images].mean(1), n_col)
            im = axes[1,0].imshow(pred_tiled)
            plt.colorbar(im, ax=axes[1,0])
            axes[1,0].set_title("predicted")        
        
            diff = pred_tiled-features_tiled
            diff_max = max(abs(diff.min()), diff.max())
            im = axes[2,0].imshow(diff, cmap="seismic", vmin=-diff_max, vmax=diff_max)
            plt.colorbar(im, ax=axes[2,0])
            axes[2,0].set_title("diff")
    
    if hasattr(model, 'get_suppl'):        
        suppl_dict = model.get_suppl(colored=True)
        if 'images' in suppl_dict:
            suppl_images = suppl_dict['images']
            fig, axes = plt.subplots(1, len(suppl_images), figsize=(4*len(suppl_images), 3), squeeze=False)
            for i, (key, (img, norm, cmap)) in enumerate(suppl_images.items()):
                im = axes[0, i].imshow(img)
                plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[0, i])
                axes[0, i].set_title(key)
        if 'plots' in suppl_dict:
            suppl_plots = suppl_dict['plots']
            for key, val in suppl_plots.items():
                fig, ax = plt.subplots(1, 1, **val['kwargs'])
                val['plot'](ax)
        
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