import numpy as np
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import math

class SimulatedPSFDataset(Dataset):
    """
    Base class.
    """
    def __init__(self, out_size=(32, 32), length=512, dropout_p=0, random_z=False, psf_params={'A':[500,2000], 'bg':[0,100]},
                 noise_params={'poisson':True, 'gaussian':10}, normalize=True, padding=[8,8], *args, **kwargs):
        Dataset.__init__(self)
        
        self.padding = padding # x, y
        self.gen_size = (out_size[0]+2*self.padding[0], out_size[1]+2*self.padding[1])
        self.out_size = out_size
        
        if isinstance(length, int):
            output_psfs_shape = [length, 1]
        else:
            output_psfs_shape = length
            
        if not 'A' in psf_params:
            print("psf amplitude defaulting to [500, 2000]")
            A = [500, 2000]
        else:
            A = psf_params['A']
            
        if not 'bg' in psf_params:
            print("psf bg defaulting to [0, 100]")
            bg = [0, 100]
        else:
            bg = psf_params['bg']
        
        self.random_z = random_z
        self.shifts = list([np.random.uniform(-out_size[0]/3+self.padding[0], out_size[0]/3-self.padding[0], np.prod(output_psfs_shape)),
                            np.random.uniform(-out_size[1]/3+self.padding[1], out_size[1]/3-self.padding[1], np.prod(output_psfs_shape)),                                
                            ])
        if self.random_z:
            self.shifts.append(np.random.uniform(-2*np.pi, 2*np.pi, np.prod(output_psfs_shape)))
        self.shifts = np.stack(self.shifts, axis=-1)
            
        # print(np.prod(output_psfs_shape))
        psfs = self.generate_psfs(self.gen_size, np.prod(output_psfs_shape), self.shifts, psf_params, *args, **kwargs) #assumes normalized from 0 to 1
        
        if dropout_p > 0:
            psfs = psfs * (np.random.rand(psfs.shape[0], 1, 1) > dropout_p)
        
        psfs = psfs * np.random.uniform(A[0], A[1], psfs.shape[0])[:, None, None]
        
        psfs = psfs.reshape(output_psfs_shape[0], output_psfs_shape[1], psfs.shape[1], psfs.shape[2])
        psfs = psfs.sum(axis=1)
        
        psfs = psfs + np.random.uniform(bg[0], bg[1], output_psfs_shape[0])[:, None, None]
        # print(psfs.dtype)
        
        if len(noise_params) > 0:
            psfs = self.add_noise(psfs, noise_params)

        if normalize:
            psfs -= psfs.min(axis=(1,2), keepdims=True)
            psfs /= psfs.max(axis=(1,2), keepdims=True)
            
        self.psfs = psfs.astype(np.float32)[:, None, ...]
        
    def generate_psfs(self, size, length, shifts, psf_params, *args, **kwargs):
        # return np.random.uniform(0, 1, (length,) + size)
        raise NotImplementedError()
        
    def add_noise(self, images, noise_params):
        ret = images
        if noise_params.pop('poisson', False):
            ret = np.random.poisson(images)
        if 'gaussian' in noise_params:
            ret = np.random.normal(images, noise_params['gaussian'])
        return ret
    
    def __len__(self):
        return self.psfs.shape[0]

    def __getitem__(self, key):
        shift = [np.random.randint(0, 2*i+1) for i in self.padding]
        if self.random_z:
            shift.append(0)
        return self.psfs[key][:,shift[0]:shift[0]+self.out_size[0],shift[1]:shift[1]+self.out_size[1]], self.shifts[key] + shift # have not checked shifts
        

class Gaussian2DPSFDataset(SimulatedPSFDataset):
    def __init__(self, out_size=(32, 32), length=512, dropout_p=0, psf_params={'A':[500,2000], 'bg':[0,100], 'sig_x':[5,5], 'sig_y':[5,5]},
                 noise_params={'poisson':True, 'gaussian':100}, *args, **kwargs):
        SimulatedPSFDataset.__init__(self, out_size=out_size, length=length,
                                     dropout_p=dropout_p, random_z=False,
                                     psf_params=psf_params, noise_params=noise_params, *args, **kwargs)
        
    def generate_psfs(self, size, length, shifts, psf_params, *args, **kwargs):
        xs = np.arange(0, size[0]) - 0.5*(size[0]-1)
        ys = np.arange(0, size[1]) - 0.5*(size[1]-1)
        XS, YS = np.meshgrid(xs, ys, indexing='ij')
        # print(shifts.shape)
        # print(psf_params['sig_x'].dtype)
        ret = np.exp(-((XS[None,...]-shifts[:,0,None,None])**2/(2*np.random.uniform(*psf_params['sig_x'], (length, 1,1))) \
                       + (YS[None,...]-shifts[:,1,None,None])**2/(2*np.random.uniform(*psf_params['sig_x'], (length, 1,1)))))
        ret -= ret.min(axis=(1,2), keepdims=True)
        ret /= ret.max(axis=(1,2), keepdims=True)
        return ret
    
    
class FourierOpticsPSFDataset(SimulatedPSFDataset):
    def __init__(self, out_size=(32, 32), length=512, dropout_p=0, random_z=False, psf_params={'A':[500,2000], 'bg':[0,100], 'sig_x':[2.5,2.5], 'sig_y':[2.5,2.5]},
                 psf_zerns={},
                 noise_params={'poisson':True, 'gaussian':100},
                *args, **kwargs):
        
        SimulatedPSFDataset.__init__(self, out_size=out_size, length=length, dropout_p=dropout_p,
                                     random_z=random_z, psf_params=psf_params,
                                     noise_params=noise_params, psf_zerns=psf_zerns, *args, **kwargs)
        
        
    def generate_psfs(self, size, length, shifts, psf_params, *args, **kwargs):        
        kx = np.fft.fftshift(np.fft.fftfreq(4*size[0]))
        ky = np.fft.fftshift(np.fft.fftfreq(4*size[1]))
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')
        
        us = np.linspace(-5, 5, 4*size[0])
        vs = np.linspace(-5, 5, 4*size[1])
        US, VS = np.meshgrid(us, vs, indexing='ij')
        R = np.sqrt(US**2 + VS**2)
        pupil_mag = np.sqrt(1-np.minimum(R, 1)**2)
        # pupil_phase = np.zeros(pupil_mag.shape)
        pupil_phase = self.calculate_pupil_phase(R*(R<=1), np.arctan2(US, VS), kwargs.pop("psf_zerns", {}))
        
        self.pupil = pupil_mag * np.exp(1j*pupil_phase)        
        self.pupil = self.pupil[int(1.5*size[0]):int(-1.5*size[0]), int(1.5*size[1]):int(-1.5*size[1])]
        
        shifted_pupil_phase = np.tile(pupil_phase, (shifts.shape[0], 1, 1))
        shifted_pupil_phase = shifted_pupil_phase - 2 * np.pi * (self.KX[None,...] * shifts[:,0,None,None])
        shifted_pupil_phase = shifted_pupil_phase - 2 * np.pi * (self.KY[None,...] * shifts[:,1,None,None])
        if self.random_z:
            shifted_pupil_phase = shifted_pupil_phase + np.sqrt(1-np.minimum(R, 1)**2) * shifts[:,2,None,None]        
        
        # pupil = np.fft.fftshift(pupil)
        # shifted_pupils = pupil[None,...]*np.exp(-2j*np.pi*(self.kx*shifts[:,0,None,None] + self.ky*shifts[:,1,None,None]))
        # # shifted_pupils = np.stack([pupil,]*length)
        shifted_pupils = pupil_mag[None,...]*np.exp(1j*shifted_pupil_phase)
        
        psfs = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(shifted_pupils)))
        psfs = psfs[:, int(1.5*size[0]):int(-1.5*size[0]), int(1.5*size[1]):int(-1.5*size[1])]
        psfs = np.abs(psfs)**2
        
        psfs -= psfs.min(axis=(1,2), keepdims=True)
        psfs /= psfs.max(axis=(1,2), keepdims=True)

        return psfs
    
    def calculate_pupil_phase(self, radial_distance, azimuthal_angle, zernikes):
        pupil_phase = np.zeros(radial_distance.shape)
        for key, val in zernikes.items():
            pupil_phase += val * calculate_zernike(key, radial_distance, azimuthal_angle)
            
        return pupil_phase


class FourierOptics2DPSFDataset(FourierOpticsPSFDataset):
    def __init__(self, *args, **kwargs):
        FourierOpticsPSFDataset.__init__(self, random_z=False, *args, **kwargs)
        
class FourierOptics3DPSFDataset(FourierOpticsPSFDataset):
    def __init__(self, *args, **kwargs):
        FourierOpticsPSFDataset.__init__(self, random_z=True, *args, **kwargs)
    
def calculate_zernike(j, radial_distance, azimuthal_angle):
    def decode_ansi_index(j):
        n = np.round(np.sqrt(j*2+1)-1).astype(np.int)
        m = 2*j - n*(n+2)
        return m, n
    
    def calculate_radial_polynomial(m, n, rho):
        if (n-m) % 2 == 0:
            ret = 0
            for k in np.arange(0, (n-m)//2+1):
                ret += np.power(-1, k) * math.comb(n-k, k) * math.comb(n-2*k, (n-m)//2-k) * np.power(rho, n-2*k)
            return ret
        else:
            return np.zeros(rho.shape)        
        
    m, n = decode_ansi_index(j)
    # print(m,n)
        
    z = calculate_radial_polynomial(np.abs(m), n, radial_distance)
    
    if m >=  0:
        z *= np.cos(m * azimuthal_angle)
    else:
        z *= np.sin(np.abs(m) * azimuthal_angle)
    
    return z
        

def inspect_psfs(psf_dataset, indices=None):
    if indices is None:
        indices = np.random.choice(psf_dataset.psfs.shape[0], 5, replace=False)
    fig, axes = plt.subplots(1, len(indices), figsize=(4*len(indices), 3))
    for i, val in enumerate(indices):
        im = axes[i].imshow(psf_dataset[val][0][0])
        plt.colorbar(im, ax=axes[i])
        axes[i].set_title("id: {}".format(val))
    if hasattr(psf_dataset, 'pupil'):
        fig, axes = plt.subplots(1, 2, figsize=(4*2, 3))
        im = axes[0].imshow(np.abs(psf_dataset.pupil))
        plt.colorbar(im, ax=axes[0])
        im = axes[1].imshow(np.angle(psf_dataset.pupil))
        plt.colorbar(im, ax=axes[1])

