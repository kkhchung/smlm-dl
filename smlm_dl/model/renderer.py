import functools
import numpy as np
import torch
from torch import nn

import util, zernike
from . import base


class BaseRendererModel(base.BaseModel):
    """
    Base renderer class for a few standardized output functions
    """
    params_ref = dict()
    params_default = dict()
    params_ind = dict()
    
    def __init__(self, img_size, fit_params):
        super().__init__()
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


class Gaussian2DRenderer(BaseRendererModel):
    
    def __init__(self, img_size, fit_params):
        
        super().__init__(img_size, fit_params)
        
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


class Template2DRenderer(BaseRendererModel):
    def __init__(self, img_size, fit_params, template_init=None, template_padding=None, conv=None):
        # The 2x sampling avoids the banding artifact that is probably caused by FFT subpixels shifts
        # This avoids any filtering in the spatial / fourier domain

        super().__init__(img_size, fit_params)
        
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
        self.template = nn.Sequential(base.ParameterModule(template + noise),
                                      nn.GELU(),
                                      # nn.Dropout2d(p=0.25),
                                     )
        
        self.template_pooling = nn.AvgPool2d((scale_factor,scale_factor), (scale_factor,scale_factor), padding=0)
        
        kx = torch.fft.fftshift(torch.fft.fftfreq(template_size_scaled[0]*2))
        ky = torch.fft.fftshift(torch.fft.fftfreq(template_size_scaled[1]*2))
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        self.register_buffer('KX', KX, False)
        self.register_buffer('KY', KY, False)
        
        if not conv is None:
            self.conv = nn.Sequential(base.ParameterModule(torch.ones(1, 1, conv, conv) / conv**2),
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
    
    def get_suppl(self, colored=False):
        res = {'images': {},
               }
        template = self._calculate_template(True).detach().numpy()
        template_2x = self._calculate_template(False).detach().numpy()
        if colored:
            template = util.color_images(template, full_output=True)
            template_2x = util.color_images(template_2x, full_output=True)
        res['images'].update({'template':template, 'template 2x':template_2x, })
        
        if not self.conv is None:
            conv_kernel = self.conv(None).detach().numpy()[0,0]
            conv_kernel = util.color_images(conv_kernel, full_output=True)
            res['images']['conv'] = conv_kernel
        return res


class FourierOptics2DRenderer(BaseRendererModel):
    def __init__(self, img_size, fit_params,
                 pupil_params={'scale':0.75, 'apod':False, 'phase_init_zern':{}, 'remove_tilt_tip_defocus':False},):

        super().__init__(img_size, fit_params)
        
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
                base.ParameterModule(pupil_magnitude),
                nn.ReLU(),
                # nn.Dropout(p=0.25),
            )
        else:
            self.pupil_magnitude = pupil_magnitude
            
        pupil_phase = torch.zeros((self.img_size[0], self.img_size[1]))
        nn.init.xavier_normal_(pupil_phase, 0.1)
        pupil_phase = pupil_phase + zernike.calculate_pupil_phase(R, torch.atan2(US, VS), pupil_params.get("phase_init_zern", {}))
        self.pupil_phase = nn.Sequential(
            base.ParameterModule(pupil_phase),
            # nn.Identity(),
            # nn.Hardtanh(-np.pi, np.pi),
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
                base.ParameterModule(torch.ones(1)*0.1),
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
        
        self.remove_tilt_tip_defocus = pupil_params["remove_tilt_tip_defocus"]
    
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
        if self.remove_tilt_tip_defocus:
            pupil_phase = self._substract_tilt_tip_defocus(pupil_phase)
        pupil_prop = self.pupil_prop
        return pupil_magnitude, pupil_phase, pupil_prop
    
    def _substract_tilt_tip_defocus(self, pupil_phase):
        pupil_phase_masked = restoration.unwrap_phase(np.fmod(np.ma.array(pupil_phase.detach().numpy(), mask=~self.mask), np.pi))
        diff = pupil_phase_masked.filled(0) - pupil_phase.detach().numpy()
        diff[~self.mask] = 0
        ret = pupil_phase + torch.as_tensor(diff, dtype=torch.float)
        pupil_phase_masked = torch.as_tensor(pupil_phase_masked.data[self.mask], dtype=torch.float)
        for basis in [self.zern_defocus, self.zern_tip, self.zern_tilt]:
            res = torch.linalg.lstsq(basis[self.mask].reshape(-1, 1), pupil_phase_masked.reshape(-1, 1))[0]
            pred = basis * res
            ret = ret - pred * self.mask
        return ret
    
    def get_feedback(self):
        return torch.stack(self._calculate_pupil()).unsqueeze(0)
    
    def get_suppl(self, colored=False):
        pupil_magnitude, pupil_phase, pupil_prop = self._calculate_pupil()
        pupil_magnitude = pupil_magnitude.detach().numpy()
        pupil_phase = pupil_phase.detach().numpy()
        pupil_prop = pupil_prop.detach().numpy()
        
        zern_coeffs = zernike.fit_zernike_from_pupil(pupil_magnitude*np.exp(1j*pupil_phase), 16, self.R, np.arctan2(self.US, self.VS))
        zern_plot = functools.partial(zernike.plot_zernike_coeffs, zernike_coeffs=zern_coeffs)
        
        if colored:
            pupil_magnitude = util.color_images(pupil_magnitude, full_output=True)
            pupil_phase = util.color_images(pupil_phase, vsym=True, full_output=True)
            pupil_prop = util.color_images(pupil_prop, full_output=True)
            
        return {'images':{'pupil mag':pupil_magnitude, 'pupil phase':pupil_phase, 'z propagate':pupil_prop},
                'plots':{'zernike': {'plot':zern_plot, 'kwargs':{'figsize':(8,3)}}},
               }