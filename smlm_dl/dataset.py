import numpy as np
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import math

class SimulatedPSFDataset(Dataset):
    """
    Base class.
    """
    def __init__(self, size=(32, 32), length=512, psf_params={'A':[500,2000], 'bg':[0,100]}, noise_params={'poisson':True, 'gaussian':100}, *args, **kwargs):
        Dataset.__init__(self)
        
        self.shifts = np.stack([np.random.uniform(-size[0]//2, size[0]//2, length),
                                np.random.uniform(-size[1]//2, size[1]//2, length),
                                ], axis=-1)
        
        psfs = self.generate_psfs(size, length, self.shifts, psf_params, **kwargs) #assumes normalized from 0 to 1
        
        if not 'A' in psf_params:
            print("psf amplitude defaulting to [500, 2000]")
            A = [500, 2000]
        else:
            A = psf_params['A']
            
        if not 'bg' in psf_params:
            print("psf bg defaulting to [0, 100]")
            bg = [0, 100]
        else:
            bg = psf_params['A']
            
        psfs = psfs * np.random.uniform(A[0], A[1], length)[:, None, None]
        psfs = psfs + np.random.uniform(bg[0], bg[1], length)[:, None, None]
        # print(psfs.dtype)
        
        if len(noise_params) > 0:
            psfs = self.add_noise(psfs, noise_params)
            
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
        return self.psfs[key], self.shifts[key]
        

class Gaussian2DPSFDataset(SimulatedPSFDataset):
    def __init__(self, size=(32, 32), length=512, psf_params={'A':[500,2000], 'bg':[0,100], 'sig_x':[5,5], 'sig_y':[5,5]},
                 noise_params={'poisson':True, 'gaussian':100}, *args, **kwargs):
        SimulatedPSFDataset.__init__(self, size, length, psf_params, noise_params, *args, **kwargs)
        
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
    

class FourierOptics2DPSFDataset(SimulatedPSFDataset):
    def __init__(self, size=(32, 32), length=512, psf_params={'A':[500,2000], 'bg':[0,100], 'sig_x':[2.5,2.5], 'sig_y':[2.5,2.5]},
                 psf_zerns={},
                 noise_params={'poisson':True, 'gaussian':100},
                *args, **kwargs):
        
        kx = np.fft.fftfreq(4*size[0])
        ky = np.fft.fftfreq(4*size[1])
        self.kx, self.ky = np.meshgrid(kx, ky, indexing='ij')
        
        SimulatedPSFDataset.__init__(self, size, length, psf_params, noise_params, **{"psf_zerns":psf_zerns})
        
        
    def generate_psfs(self, size, length, shifts, psf_params, *args, **kwargs):
        us = np.linspace(-5, 5, 4*size[0])
        vs = np.linspace(-5, 5, 4*size[1])
        US, VS = np.meshgrid(us, vs, indexing='ij')
        R = np.sqrt(US**2 + VS**2)
        pupil_mag = np.sqrt(1-np.minimum(R, 1)**2)
        # pupil_phase = np.zeros(pupil_mag.shape)
        pupil_phase = self.calculate_pupil_phase(R*(R<=1), np.arctan2(US, VS), kwargs.pop("psf_zerns", {}))
        pupil = pupil_mag * np.exp(1j*pupil_phase)
        
        self.pupil = pupil[int(1.5*size[0]):int(-1.5*size[0]), int(1.5*size[1]):int(-1.5*size[1])]
        
        pupil = np.fft.fftshift(pupil)
        shifted_pupils = pupil[None,...]*np.exp(-2j*np.pi*(self.kx*shifts[:,0,None,None] + self.ky*shifts[:,1,None,None]))
        # shifted_pupils = np.stack([pupil,]*length)
        
        psfs = np.fft.fftshift(np.fft.ifft2(shifted_pupils))
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
        im = axes[i].imshow(psf_dataset.psfs[val, 0])
        plt.colorbar(im, ax=axes[i])
        axes[i].set_title("id: {}".format(val))
    if hasattr(psf_dataset, 'pupil'):
        fig, axes = plt.subplots(1, 2, figsize=(4*2, 3))
        im = axes[0].imshow(np.abs(psf_dataset.pupil))
        plt.colorbar(im, ax=axes[0])
        im = axes[1].imshow(np.angle(psf_dataset.pupil))
        plt.colorbar(im, ax=axes[1])

