import numpy as np
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import enum
import scipy
from scipy import ndimage, signal
import zernike

@enum.unique
class Augmentation(enum.Enum):
    PIXEL_SHIFT = 1
    NOISE_GAUSSIAN =2

class SimulatedPSFDataset(Dataset):
    """
    Base class.
    """
    def __init__(self, out_size=(32, 32), length=512, dropout_p=0, random_z=False, psf_params={'A':[500,2000], 'bg':[0,100]},
                 noise_params={'poisson':True, 'gaussian':10}, normalize=True, augmentations={Augmentation.PIXEL_SHIFT:[8,8]}, *args, **kwargs):
        Dataset.__init__(self)
        
        for key in augmentations:
            if not isinstance(key, Augmentation):
                raise Exception("Augmentation '{}' not recognized. Use Augmentation enum.".format(key))
        self.augmentations = augmentations
        self.padding = augmentations.get(Augmentation.PIXEL_SHIFT, [0,0]) # x, y
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
        self.shifts = list([np.random.uniform(-out_size[0]/3+self.padding[0]/2, out_size[0]/3-self.padding[0]/2, np.prod(output_psfs_shape)),
                            np.random.uniform(-out_size[1]/3+self.padding[1]/2, out_size[1]/3-self.padding[1]/2, np.prod(output_psfs_shape)),                                
                            ])
        if self.random_z:
            self.shifts.append(np.random.uniform(-10, 10, np.prod(output_psfs_shape)))
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
            
        self.images = psfs.astype(np.float32)[:, None, ...]
        
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
        return self.images.shape[0]

    def __getitem__(self, key):
        image = self.images[key]
        label = self.shifts[key]
        
        if Augmentation.PIXEL_SHIFT in self.augmentations:
            shift = [np.random.randint(0, 2*i+1) for i in self.padding]
            if self.random_z:
                shift.append(0)
            image = image[:,shift[0]:shift[0]+self.out_size[0],shift[1]:shift[1]+self.out_size[1]]
            label = label + shift
            
        if Augmentation.NOISE_GAUSSIAN in self.augmentations:
            noise_sig = self.augmentations[Augmentation.NOISE_GAUSSIAN] * (image.max() - image.min())
            image = np.random.normal(image, noise_sig).astype(np.float32)
            
        return image, label
        

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
    def __init__(self, out_size=(32, 32), length=512, dropout_p=0, random_z=False,
                 psf_params={'A':[500,2000], 'bg':[0,100], 'sig_x':[2.5,2.5], 'sig_y':[2.5,2.5],
                            'apod':False, 'pupil_scale':0.75},
                 psf_zerns={},
                 noise_params={'poisson':True, 'gaussian':100},
                *args, **kwargs):
        
        SimulatedPSFDataset.__init__(self, out_size=out_size, length=length, dropout_p=dropout_p,
                                     random_z=random_z, psf_params=psf_params,
                                     noise_params=noise_params, psf_zerns=psf_zerns, *args, **kwargs)
        
        
    def generate_psfs(self, size, length, shifts, psf_params, *args, **kwargs):
        pupil_padding_factor = 4
        pupil_padding_clip = 0.5 * (pupil_padding_factor - 1)
        kx = np.fft.fftshift(np.fft.fftfreq(pupil_padding_factor*size[0]))
        ky = np.fft.fftshift(np.fft.fftfreq(pupil_padding_factor*size[1]))
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')
        
        us = np.linspace(-1, 1, pupil_padding_factor*size[0]) *pupil_padding_factor / psf_params.get('pupil_scale', 0.75)
        vs = np.linspace(-1, 1, pupil_padding_factor*size[1]) *pupil_padding_factor / psf_params.get('pupil_scale', 0.75)
        US, VS = np.meshgrid(us, vs, indexing='ij')
        R = np.sqrt(US**2 + VS**2)
        
        if psf_params.get('apod', False):
            pupil_mag = np.sqrt(1-np.minimum(R, 1)**2)
        else:
            pupil_mag = (R <= 1).astype(np.float)
        pupil_phase = zernike.calculate_pupil_phase(R*(R<=1), np.arctan2(US, VS), kwargs.get("psf_zerns", {}))
        
        self.pupil = pupil_mag * np.exp(1j*pupil_phase)        
        self.pupil = self.pupil[int(pupil_padding_clip*size[0]):int(-pupil_padding_clip*size[0]), int(pupil_padding_clip*size[1]):int(-pupil_padding_clip*size[1])]
        
        shifted_pupil_phase = np.tile(pupil_phase, (shifts.shape[0], 1, 1))
        shifted_pupil_phase = shifted_pupil_phase - 2 * np.pi * (self.KX[None,...] * shifts[:,0,None,None])
        shifted_pupil_phase = shifted_pupil_phase - 2 * np.pi * (self.KY[None,...] * shifts[:,1,None,None])
        if self.random_z:
            shifted_pupil_phase = shifted_pupil_phase + np.sqrt(1-np.minimum(R, 1)**2) * shifts[:,2,None,None]        
        
        shifted_pupils = pupil_mag[None,...]*np.exp(1j*shifted_pupil_phase)
        
        psfs = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(shifted_pupils)))
        psfs = psfs[:, int(pupil_padding_clip*size[0]):int(-pupil_padding_clip*size[0]), int(pupil_padding_clip*size[1]):int(-pupil_padding_clip*size[1])]
        psfs = np.abs(psfs)**2
        
        psfs -= psfs.min(axis=(1,2), keepdims=True)
        psfs /= psfs.max(axis=(1,2), keepdims=True)

        return psfs


class FourierOptics2DPSFDataset(FourierOpticsPSFDataset):
    def __init__(self, *args, **kwargs):
        FourierOpticsPSFDataset.__init__(self, random_z=False, *args, **kwargs)
        
class FourierOptics3DPSFDataset(FourierOpticsPSFDataset):
    def __init__(self, *args, **kwargs):
        FourierOpticsPSFDataset.__init__(self, random_z=True, *args, **kwargs)
    

def inspect_images(dataset, indices=None):
    if indices is None:
        indices = np.random.choice(dataset.images.shape[0], 5, replace=False)
    fig, axes = plt.subplots(1, len(indices), figsize=(4*len(indices), 3))
    for i, val in enumerate(indices):
        im = axes[i].imshow(dataset[val][0][0])
        plt.colorbar(im, ax=axes[i])
        axes[i].set_title("id: {}".format(val))
    fig, axes = plt.subplots(1, len(indices), figsize=(4*len(indices), 3))
    for i, val in enumerate(indices):
        im = axes[i].imshow(np.log(dataset[val][0][0]))
        plt.colorbar(im, ax=axes[i])
        axes[i].set_title("id: {}".format(val))
    if hasattr(dataset, 'pupil'):
        fig, axes = plt.subplots(1, 2, figsize=(4*2, 3))
        im = axes[0].imshow(np.abs(dataset.pupil))
        plt.colorbar(im, ax=axes[0])
        im = axes[1].imshow(np.angle(dataset.pupil))
        plt.colorbar(im, ax=axes[1])


class SingleImageDataset(Dataset):
    """
    Repeatedly sample a single image.
    """
    def __init__(self, data, out_size=(64, 64), length=16, img_params={'A':[0.5,2.0], 'bg':[0,10], 'shifts':[5, 5], 'conv':np.ones((3,3))},
                 noise_params={'poisson':True, 'gaussian':10}, normalize=True, augmentations={Augmentation.PIXEL_SHIFT:[8,8]}, *args, **kwargs):
        Dataset.__init__(self)
        
        for key in augmentations:
            if not isinstance(key, Augmentation):
                raise Exception("Augmentation '{}' not recognized. Use Augmentation enum.".format(key))
        self.augmentations = augmentations
        self.padding = augmentations.get(Augmentation.PIXEL_SHIFT, [0,0]) # x, y
        self.gen_size = (out_size[0]+2*self.padding[0], out_size[1]+2*self.padding[1])
        self.out_size = out_size
                   
        A = img_params.get('A', [1,1])        
        bg = img_params.get('bg', [0,0])
        shifts = img_params.get('shifts', [out_size[0]/3-self.padding[0]/2, out_size[1]/3-self.padding[1]/2])
        conv = img_params.get('conv', None)
        
        self.shifts = list([np.random.uniform(-shifts[0], shifts[0], length),
                            np.random.uniform(-shifts[1], shifts[1], length),                                
                            ])
        self.shifts = np.stack(self.shifts, axis=-1)
            
        images = self.generate_images(data, self.gen_size, length, conv, self.shifts, *args, **kwargs)
        
        images = images * np.random.uniform(A[0], A[1], length)[:, None, None]
        images = images + np.random.uniform(bg[0], bg[1], length)[:, None, None]
        
        if len(noise_params) > 0:
            images = self.add_noise(images, noise_params)

        if normalize:
            images -= images.min(axis=(1,2), keepdims=True)
            images /= images.max(axis=(1,2), keepdims=True)
            
        self.images = images.astype(np.float32)[:, None, ...]
        
    def generate_images(self, data, size, length, conv, shifts, *args, **kwargs):
        # add padding larger than shifts
        shift_max = [np.ceil(np.max([np.abs(shifts[:,i].min()), shifts[:,i].max()])).astype(int) for i in range(len(shifts.shape))]
        crop_size = [size[i] + 2*shift_max[i] for i in range(len(data.shape))]
        data = data[:crop_size[0],:crop_size[1]]
        if not conv is None:
            data = ndimage.convolve(data, conv)
        
        # zero padding for fft
        padding = [(int(np.ceil(1.5 * data.shape[0])),)*2, (int(np.ceil(1.5 * data.shape[1])),)*2]
        data = np.pad(data, padding, mode='wrap')
                
        kx = np.fft.fftshift(np.fft.fftfreq(data.shape[0]))
        ky = np.fft.fftshift(np.fft.fftfreq(data.shape[1]))
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')
        
        fft_image = np.fft.fftshift(np.fft.fft2(data))
        fft_image_mag = np.abs(fft_image)
        fft_image_phase = np.angle(fft_image)
        
        # helps remove ringing artifacts
        fft_image_mag = fft_image_mag * signal.windows.tukey(fft_image_mag.shape[0], alpha=0.5)[:,None]
        fft_image_mag = fft_image_mag * signal.windows.tukey(fft_image_mag.shape[1], alpha=0.5)[None,:]
        
        # x, y shift
        fft_image_phase = fft_image_phase - 2 * np.pi * (self.KX[None,...] * shifts[:,0,None,None])
        fft_image_phase = fft_image_phase - 2 * np.pi * (self.KY[None,...] * shifts[:,1,None,None])
        
        shifted_fft = fft_image_mag * np.exp(1j * fft_image_phase)
        shifted_img = np.fft.ifft2(np.fft.ifftshift(shifted_fft))
        
        crop = np.concatenate([shift_max[i] + padding[i] for i in range(len(data.shape))])        
        shifted_img = shifted_img[:, crop[0]:-crop[1], crop[2]:-crop[3]]
        
        return np.abs(shifted_img)
        
    def add_noise(self, images, noise_params):
        ret = images
        if noise_params.pop('poisson', False):
            ret = np.random.poisson(images)
        if 'gaussian' in noise_params:
            ret = np.random.normal(images, noise_params['gaussian'])
        return ret
    
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, key):
        image = self.images[key]
        label = self.shifts[key]
        
        if Augmentation.PIXEL_SHIFT in self.augmentations:
            shift = [np.random.randint(0, 2*i+1) for i in self.padding]
            image = image[:,shift[0]:shift[0]+self.out_size[0],shift[1]:shift[1]+self.out_size[1]]
            label = label + shift
            
        if Augmentation.NOISE_GAUSSIAN in self.augmentations:
            noise_sig = self.augmentations[Augmentation.NOISE_GAUSSIAN] * (image.max() - image.min())
            image = np.random.normal(image, noise_sig).astype(np.float32)
            
        return image, label