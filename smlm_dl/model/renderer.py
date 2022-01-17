import functools
import numpy as np
import torch
from torch import nn

from tqdm.auto import tqdm, trange

import simulate, spline, util, zernike
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
    
    
class Spline2DRenderer(BaseRendererModel):
    
    def __init__(self, img_size, fit_params, k=3, template_init=None):
        super().__init__(img_size, fit_params)
        
        self.k = k
        self.out_size = img_size
        if template_init is None:
            template_init = simulate.simulate_centered_2d_gaussian(img_size[0], img_size[1])
            # noise = torch.empty(template_init.shape)
            # nn.init.xavier_uniform_(noise, 0.1)
            # template_init = template_init + noise
        
        coeffs = torch.tensor(template_init, dtype=torch.float)
        self.template_size = coeffs.shape
        
        self.padding = 4
        coeffs = torch.nn.functional.pad(coeffs, (self.padding,)*4)
        self.coeffs = nn.Parameter(coeffs)
        self.offset = self.padding - (k+1)//2
        self.centering_offset = [(self.out_size[0] - self.template_size[0])//2,
                                 (self.out_size[1] - self.template_size[1])//2,]
        if (any(val<0 for val in self.centering_offset)):
            raise Exception("Out_size must be larger than size of image. Limitation of using torch.scatter.")
            
        index_a = torch.arange(self.template_size[1]+1).tile(self.template_size[0]+1, 1)
        index_b = torch.arange(self.out_size[0]+2)[:,None].tile(1, self.out_size[1]+2)
        self.register_buffer('index_a', index_a)
        self.register_buffer('index_b', index_b)
        
        if not template_init is None:
            self.fit_image(template_init)
        
    def _render_images(self, mapped_params, batch_size=None, raw=False):
        shift_pixel = {key: torch.floor(mapped_params[key]/1).type(torch.int) for key in ['x', 'y']}
        shifts_subpixel = {key: torch.remainder(mapped_params[key], 1) for key in ['x', 'y']}
        
        template = self._calculate_template(shifts_subpixel)
        if raw:
            return template
        
        x_a = torch.zeros((shift_pixel['y'].shape[0], shift_pixel['y'].shape[1], self.out_size[0]+2, self.out_size[1]+2))
        
        index = self.index_a.tile(shift_pixel['y'].shape[0], shift_pixel['y'].shape[1], 1, 1) \
                + shift_pixel['y'] \
                + self.centering_offset[1] + 1
        index = torch.clamp(index, 0, x_a.shape[3]-1)
        # print(x_a.shape, index.shape, template.shape)
        x_a.scatter_(3, index, template)
        
        x_b = torch.zeros_like(x_a)
        
        index = self.index_b.tile(shift_pixel['x'].shape[0], shift_pixel['x'].shape[1], 1, 1) \
                + shift_pixel['x'] \
                + self.centering_offset[0] + 1
        index = torch.clamp(index, 0, x_b.shape[2]-1)
        # print(x_b.shape, index.shape, template.shape)
        x_b.scatter_(2, index, x_a)
        
        x = x_b[:,:,1:-1,1:-1]
        
        x = x * mapped_params['A'] + mapped_params['bg']
        
        return x
    
    def _calculate_template(self, shifts_subpixel):
        bases = {key: spline.calculate_bspline_basis(val, self.k) for key, val in shifts_subpixel.items()}
        x = 0
        for i, cx in enumerate(bases['x']):
            for j, cy in enumerate(bases['y']):
                coeffs = self.coeffs[i+self.offset:i+self.offset+self.template_size[0]+1,
                                     j+self.offset:j+self.offset+self.template_size[1]+1]
                x = x + cx * cy * coeffs
        return x
    
    def render(self, scale=1):
        x = torch.arange(0, 1, 1/scale)
        y = torch.arange(0, 1, 1/scale)
        xs, ys = torch.meshgrid(x, y, indexing='ij')

        x = {'x': xs.flatten()[:,None,None,None],
             'y': ys.flatten()[:,None,None,None],}
        
        interpolated_images = self._calculate_template(x)
        image_highres = np.empty(((self.template_size[0]+1)*scale, (self.template_size[1]+1)*scale))
        
        for i, img in enumerate(interpolated_images):
            c = i // scale
            r = i % scale
            image_highres[scale-c-1::scale, scale-r-1::scale] = interpolated_images[i,0].detach()
        
        image_highres = image_highres[scale//2:-((scale+1)//2), scale//2:-((scale+1)//2)]
        return image_highres
    
    def get_suppl(self, colored=False):
        res = {'images': {},
               }
        template = self.render(1)
        template_2x = self.render(2)
        if colored:
            template = util.color_images(template, full_output=True)
            template_2x = util.color_images(template_2x, full_output=True)
        res['images'].update({'template':template, 'template 2x':template_2x, })

        return res
    
    def fit_image(self, image_groundtruth, maxiter=200, tol=1e-9):
        image_groundtruth = torch.as_tensor(image_groundtruth).type(torch.float32)
        image_groundtruth = nn.functional.pad(image_groundtruth, (0,1,0,1))
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=10)
        
        x = {'x':torch.zeros((1,1,1,1)),
             'y':torch.zeros((1,1,1,1))}
        
        mask = torch.ones(self.coeffs.shape, dtype=torch.bool)
        mask[self.padding:-self.padding,self.padding:-self.padding] = False
        
        print("Fitting...")
        loss_log = list()
        for i in range(maxiter):
            optimizer.zero_grad()
            pred = self._render_images(x, raw=True)
            loss = loss_function(pred[0,0,:,:], image_groundtruth)
            loss_log.append(loss.detach().numpy())
            
            if (loss < tol):
                print("Early termination after {} iterations, loss tol < {} reached".format(i, tol))
                break
                
            loss.backward()
            optimizer.step()
            self.coeffs.data.masked_fill_(mask, 0)
            
        r2 = 1 - loss_log[-1] / image_groundtruth.var()
        print("Final loss: {:.3f}\tR2: {:.3f}".format(loss_log[-1], r2))
        if loss_log[-1] > 1 and r2 < 0.9:
            print("Not well fitted.")
        return loss_log
    
    
class Spline3DRenderer(BaseRendererModel):
    
    def __init__(self, img_size, fit_params, k=3, template_init=None):
        super().__init__(img_size, fit_params)
        
        self.k = k
        self.out_size = img_size
        if template_init is None:
            template_init = simulate.simulate_centered_3d_gaussian(img_size[0], img_size[1], 64)
            # noise = torch.empty(template_init.shape)
            # nn.init.xavier_uniform_(noise, 0.1)
            # template_init = template_init + noise
        
        coeffs = torch.tensor(template_init, dtype=torch.float)
        self.template_size = coeffs.shape
        
        self.padding = 4
        coeffs = torch.nn.functional.pad(coeffs, (self.padding,)*6)
        self.coeffs = nn.Parameter(coeffs)
        self.offset = self.padding - (k+1)//2
        self.centering_offset = [(self.out_size[0] - self.template_size[0])//2,
                                 (self.out_size[1] - self.template_size[1])//2,
                                 self.template_size[2]//2,
                                ]
        if (any(val<0 for val in self.centering_offset)):
            raise Exception("Out_size must be larger than size of image. Limitation of using torch.scatter.")
            
        index_a = torch.arange(self.template_size[1]+1).tile(self.template_size[0]+1, 1)
        index_b = torch.arange(self.out_size[0]+2)[:,None].tile(1, self.out_size[1]+2)
        self.register_buffer('index_a', index_a)
        self.register_buffer('index_b', index_b)
        
        if not template_init is None:
            self.spline_fit_loss_log = self.fit_image(template_init)
        
    def _render_images(self, mapped_params, batch_size=None, raw=False):
        shift_pixel = {key: torch.floor(mapped_params[key]/1).type(torch.int) for key in ['x', 'y', 'z']}
        shifts_subpixel = {key: torch.remainder(mapped_params[key], 1) for key in ['x', 'y', 'z']}
        
        template = self._calculate_template(shifts_subpixel, shift_pixel['z']+self.centering_offset[2])
        if raw:
            return template
        
        x_a = torch.zeros((shift_pixel['y'].shape[0], shift_pixel['y'].shape[1], self.out_size[0]+2, self.out_size[1]+2))
        
        index = self.index_a.tile(shift_pixel['y'].shape[0], shift_pixel['y'].shape[1], 1, 1) \
                + shift_pixel['y'] \
                + self.centering_offset[1] + 1
        index = torch.clamp(index, 0, x_a.shape[3]-1)
        x_a.scatter_(3, index, template)
        
        x_b = torch.zeros_like(x_a)
        
        index = self.index_b.tile(shift_pixel['x'].shape[0], shift_pixel['x'].shape[1], 1, 1) \
                + shift_pixel['x'] \
                + self.centering_offset[0] + 1
        index = torch.clamp(index, 0, x_b.shape[2]-1)
        x_b.scatter_(2, index, x_a)
        
        x = x_b[:,:,1:-1,1:-1]
        
        x = x * mapped_params['A'] + mapped_params['bg']
        
        return x
    
    def _calculate_template(self, shifts_subpixel, shift_pixel_z):
        index = torch.arange(4).tile(*shift_pixel_z.shape[:2], *self.coeffs.shape[:2], 1) + self.offset + shift_pixel_z.unsqueeze(-1) - 1
        index = torch.clamp(index, 0, self.coeffs.shape[2]-1)

        coeffs = self.coeffs.tile(*shift_pixel_z.shape[:2], 1, 1, 1)
        coeffs = torch.gather(coeffs, 4, index)
        
        bases = {key: spline.calculate_bspline_basis(val, self.k) for key, val in shifts_subpixel.items()}
        x = 0
        for i, cx in enumerate(bases['x']):
            for j, cy in enumerate(bases['y']):
                for k, cz in enumerate(bases['z']):
                    coeffs_tmp = coeffs[:,:,
                                        i+self.offset:i+self.offset+self.template_size[0]+1,
                                        j+self.offset:j+self.offset+self.template_size[1]+1,
                                        k]
                    x = x + cx * cy * cz * coeffs_tmp
        return x
    
    def render(self, scale=1):
        x = torch.arange(0, 1, 1/scale)
        y = torch.arange(0, 1, 1/scale)
        z = torch.arange(0, self.template_size[2], 1/scale) - self.centering_offset[2]
        
        xs, ys, zs = torch.meshgrid(x, y, z, indexing='ij')

        x = {'x': xs.reshape(-1,1,1,1),
             'y': ys.reshape(-1,1,1,1),
             'z': zs.reshape(-1,1,1,1),}
        
        interpolated_images = self._render_images(x, raw=True)
        interpolated_images = interpolated_images.reshape(scale, scale, scale*self.template_size[2], self.template_size[0]+1, self.template_size[1]+1,)
        interpolated_images = interpolated_images.moveaxis((3, 4), (0,2))
        interpolated_images = interpolated_images.reshape((self.template_size[0]+1)*scale,
                                                         (self.template_size[1]+1)*scale,
                                                         self.template_size[2]*scale)
        interpolated_images = interpolated_images[scale//2:-((scale+1)//2), scale//2:-((scale+1)//2), :]

        return interpolated_images
    
    def get_suppl(self, colored=False):
        res = {'images': {},
               }
        template = self.render(1)
        template_2x = self.render(2)
        if colored:
            template = util.color_images(template, full_output=True)
            template_2x = util.color_images(template_2x, full_output=True)
        res['images'].update({'template':template, 'template 2x':template_2x, })

        return res
    
    def fit_image(self, volume_groundtruth, maxiter=10, tol=1e-9):
        volume_groundtruth = torch.as_tensor(volume_groundtruth).type(torch.float32)
        volume_groundtruth = nn.functional.pad(volume_groundtruth, (0,0,0,1,0,1))
        
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e3, amsgrad=False)
        
        x = {'x':torch.zeros((1,1,1,1)),
             'y':torch.zeros((1,1,1,1)),
             'z':torch.arange(volume_groundtruth.shape[2]).reshape(-1,1,1,1) - self.centering_offset[2],
            }
        
        mask = torch.ones(self.coeffs.shape, dtype=torch.bool)
        mask[self.padding:-self.padding,self.padding:-self.padding] = False

        loss_log = list()
        _t = tqdm(total=maxiter)
        _t.set_description("Fitting spline ...")
        for i in range(maxiter):
            optimizer.zero_grad()
            pred = self._render_images(x, raw=True)
            pred = pred.mean(axis=1)
            pred = pred.moveaxis(0, -1)
            loss = loss_function(pred, volume_groundtruth)
            loss_log.append(loss.detach().numpy())
            _t.set_postfix(loss=loss.detach().numpy())
            _t.update(1)
            
            if (loss < tol):
                print("Early termination after {} iterations, loss tol < {} reached".format(i, tol))
                break
                
            loss.backward()
            optimizer.step()
            self.coeffs.data.masked_fill_(mask, 0)
            
        r2 = 1 - loss_log[-1] / volume_groundtruth.var()
        print("Final loss: {:.3f}\tR2: {:.3f}".format(loss_log[-1], r2))
        if loss_log[-1] > 1 and r2 < 0.9:
            print("Not well fitted.")
        return loss_log
    