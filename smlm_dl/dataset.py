import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
from matplotlib import pyplot as plt
import enum
import scipy
from scipy import ndimage, signal
import io, util, zernike
from skimage import restoration

@enum.unique
class Augmentation(enum.Enum):
    PIXEL_SHIFT = 1
    NOISE_GAUSSIAN =2
    
class ImageDataset(Dataset):
    """
    Base class.
    """
    def __init__(self, out_size=(32, 32), length=512, dropout_p=0,
                 image_params={},
                 noise_params={'poisson':True, 'gaussian':10},
                 conv_kernel=None,
                 normalize=True, augmentations={Augmentation.PIXEL_SHIFT:[8,8]},
                 image_params_preset={}):
        super().__init__()
        
        for key in augmentations:
            if not isinstance(key, Augmentation):
                raise Exception("Augmentation '{}' not recognized. Use Augmentation enum.".format(key))
        
        self.augmentations = augmentations
        self.padding = augmentations.get(Augmentation.PIXEL_SHIFT, [0,0]) # x, y
        self.gen_size = (out_size[0]+2*self.padding[0], out_size[1]+2*self.padding[1])
        self.out_size = out_size
        
        output_image_shape = np.atleast_1d(np.asarray(length))
        if output_image_shape.shape[0]<2:
            output_image_shape = np.concatenate([output_image_shape, [1]])
        
        self.set_params(output_image_shape, image_params, image_params_preset)
            
        shifts = np.stack([self.params['x'].flatten(), self.params['y'].flatten(), self.params['z'].flatten()], axis=-1)
            
        images = self.generate_images(self.gen_size, output_image_shape, shifts, image_params)
        
        if dropout_p > 0:
            images = images * (np.random.rand(images.shape[0], 1, 1) > dropout_p)
        
        images = images * self.params['A'].reshape(-1, 1, 1)
        
        images = images.reshape(output_image_shape[0], output_image_shape[1], images.shape[1], images.shape[2])
        images = images.sum(axis=1, keepdims=True)
        
        images = images + self.params['bg'].reshape(-1, 1, 1, 1)
        
        if not conv_kernel is None:
            conv_kernel = torch.as_tensor(conv_kernel, dtype=torch.float).reshape(1, 1, conv_kernel.shape[-2], conv_kernel.shape[-1])
            images = torch.as_tensor(images, dtype=torch.float)
            images = torch.nn.functional.pad(images, (conv_kernel.shape[-1]//2,)*2 + (conv_kernel.shape[-2]//2,)*2, mode="reflect")
            images = torch.nn.functional.conv2d(images, conv_kernel, padding=0).numpy()
        
        if len(noise_params) > 0:
            images = self.add_noise(images, noise_params)

        if normalize:
            images -= images.min(axis=(2,3), keepdims=True)
            images /= images.max(axis=(2,3), keepdims=True)
            
        self.images = images.astype(np.float32)
        
    def set_params(self, output_image_shape, image_params, image_params_preset):
        # print("Image parameters settings: {}".format(image_params))
        self.params = {}
        self.params['id'] = np.arange(output_image_shape[0])
        self.params['A'] = np.random.uniform(image_params['A'][0], image_params['A'][1], output_image_shape).astype(np.float32)
        self.params['bg'] = np.random.uniform(image_params['bg'][0], image_params['bg'][1], output_image_shape[0]).astype(np.float32)
        self.params['x'] = np.random.uniform(image_params['x'][0], image_params['x'][1], output_image_shape).astype(np.float32)
        self.params['y'] = np.random.uniform(image_params['y'][0], image_params['y'][1], output_image_shape).astype(np.float32)
        
        if 'z' in image_params:
            self.params['z'] = np.random.uniform(image_params['z'][0], image_params['z'][1], output_image_shape).astype(np.float32)
        else:
            self.params['z'] = np.zeros(output_image_shape).astype(np.float32)
            
        self.params.update(image_params_preset)
        
    def generate_images(self, size, length, shifts, image_params):
        raise NotImplementedError()
        
    def add_noise(self, images, noise_params):
        ret = images
        if 'poisson' in noise_params:
            ret = np.random.poisson(images)
        if 'gaussian' in noise_params:
            ret = np.random.normal(images, noise_params['gaussian'])
        return ret
    
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, key):
        image = self.images[key]
        label = {param_key: param_val[key] for param_key, param_val in self.params.items()}
        
        if Augmentation.PIXEL_SHIFT in self.augmentations:
            shift = [np.random.randint(0, 2*i+1) for i in self.padding]
            label['x'] = label['x'] - shift[0] + self.padding[0]
            label['y'] = label['y'] - shift[1] + self.padding[1]
            image = image[:,shift[0]:shift[0]+self.out_size[0],shift[1]:shift[1]+self.out_size[1]]
            
        if Augmentation.NOISE_GAUSSIAN in self.augmentations:
            noise_sig = self.augmentations[Augmentation.NOISE_GAUSSIAN] * (image.max() - image.min())
            image = np.random.normal(image, noise_sig).astype(np.float32)
            
        return image, label
    

class SingleImageDataset(ImageDataset):
    """
    Repeatedly sample a single image.
    """
    def __init__(self, data, out_size=(64, 64), length=16, dropout_p=0,
                 image_params={},
                 noise_params={'poisson':True, 'gaussian':10},
                 conv_kernel = None,
                 normalize=True, augmentations={Augmentation.PIXEL_SHIFT:[8,8]},
                 image_params_preset={}):
        
        default_image_params = {
            'A': [0.5, 2.0],
            'bg': [0, 10],
            'x': [-5, 5],
            'y': [-5, 5],
            # 'conv':np.ones((3,3)),
        }
        
        _image_params = dict(default_image_params, **image_params)
        _image_params['data'] = data
        
        super().__init__(out_size=out_size, length=length, dropout_p=dropout_p,
                         image_params=_image_params,
                         noise_params=noise_params,
                         conv_kernel=conv_kernel,
                         normalize=normalize, augmentations=augmentations,
                         image_params_preset=image_params_preset)
        
    def generate_images(self, size, length, shifts, image_params):
        data = image_params['data']
        # add padding larger than shifts
        shift_max = [np.ceil(np.max([np.abs(shifts[:,i].min()), shifts[:,i].max()])).astype(int) for i in range(len(shifts.shape))]
        crop_size = [size[i] + 2*shift_max[i] for i in range(len(data.shape))]
        data = data[:crop_size[0],:crop_size[1]]
        
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


class SimulatedPSFDataset(ImageDataset):
    def __init__(self, out_size=(32, 32), length=512, dropout_p=0,
                 image_params={},
                 noise_params={'poisson':True, 'gaussian':10},
                 normalize=True, augmentations={Augmentation.PIXEL_SHIFT:[8,8]},
                 image_params_preset={}):
        
        default_image_params = {
            'A': [500, 2000],
            'bg': [0, 100],
            'x': [-0.35*out_size[0], 0.35*out_size[0]],
            'y': [-0.35*out_size[1], 0.35*out_size[1]],
        }
        
        _image_params = dict(default_image_params, **image_params)
        
        super().__init__(out_size=out_size, length=length, dropout_p=dropout_p,
                         image_params=_image_params,
                         noise_params=noise_params,
                         normalize=normalize, augmentations=augmentations,
                         image_params_preset=image_params_preset)
        
    def generate_images(self, size, length, shifts, image_params):
        raise NotImplementedError()


class Gaussian2DPSFDataset(SimulatedPSFDataset):
    def __init__(self, out_size=(32, 32), length=512, dropout_p=0,
                 psf_params={},
                 noise_params={'poisson':True, 'gaussian':100},
                 normalize=False, augmentations={},
                 image_params_preset={}):
        
        default_image_params = {
            'sig_x':[5, 5],
            'sig_y':[5, 5],
        }
        
        _image_params = dict(default_image_params, **psf_params)

        super().__init__(out_size=out_size, length=length, dropout_p=dropout_p,
                         image_params=_image_params,
                         noise_params=noise_params,
                         normalize=normalize, augmentations=augmentations,
                         image_params_preset=image_params_preset)
        
    def generate_images(self, size, length, shifts, psf_params):
        xs = np.arange(0, size[0]) - 0.5*(size[0]-1)
        ys = np.arange(0, size[1]) - 0.5*(size[1]-1)
        XS, YS = np.meshgrid(xs, ys, indexing='ij')
        
        self.params['sig_x'] = np.random.uniform(*psf_params['sig_x'], length).astype(np.float32)
        self.params['sig_y'] = np.random.uniform(*psf_params['sig_y'], length).astype(np.float32)
        ret = np.exp(-((XS[None,...]-shifts[:,0,None,None])**2/(2*self.params['sig_x'].reshape(-1,1,1)) \
                       + (YS[None,...]-shifts[:,1,None,None])**2/(2*self.params['sig_y'].reshape(-1,1,1))))
        
        return ret
    
    
class FourierOpticsPSFDataset(SimulatedPSFDataset):
    def __init__(self, out_size=(32, 32), length=512, dropout_p=0,
                 psf_params={}, psf_zerns={},
                 noise_params={'poisson':True, 'gaussian':100},
                 normalize=False, augmentations={},
                 image_params_preset={}):
        
        default_psf_params = {
            'apod':False,
            'pupil_scale':0.75,
        }
        
        _psf_params = dict(default_psf_params, **psf_params)
        _psf_params["psf_zerns"] = psf_zerns
        
        super().__init__(out_size=out_size, length=length, dropout_p=dropout_p,
                         image_params=_psf_params,
                         noise_params=noise_params,
                         normalize=normalize, augmentations=augmentations,
                         image_params_preset=image_params_preset)
        
    def generate_images(self, size, length, shifts, psf_params):
        pupil_padding_factor = 4
        pupil_padding_clip = 0.5 * (pupil_padding_factor - 1)
        pupil_padding = int(pupil_padding_clip*size[0]), int(-pupil_padding_clip*size[0]), int(pupil_padding_clip*size[1]), int(-pupil_padding_clip*size[1])
        kx = np.fft.fftshift(np.fft.fftfreq(pupil_padding_factor*size[0]))
        ky = np.fft.fftshift(np.fft.fftfreq(pupil_padding_factor*size[1]))
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')
        
        us = np.linspace(-1, 1, pupil_padding_factor*size[0]) * (pupil_padding_factor*size[0]-1) / (size[0]-1)  / psf_params.get('pupil_scale', 0.75)
        vs = np.linspace(-1, 1, pupil_padding_factor*size[1]) * (pupil_padding_factor*size[0]-1) / (size[0]-1)  / psf_params.get('pupil_scale', 0.75)
        US, VS = np.meshgrid(us, vs, indexing='ij')
        R = np.sqrt(US**2 + VS**2)
        
        if psf_params.get('apod', False):
            pupil_mag = np.sqrt(1-np.minimum(R, 1)**2)
        else:
            pupil_mag = (R <= 1).astype(np.float)
        pupil_phase = zernike.calculate_pupil_phase(R*(R<=1), np.arctan2(US, VS), psf_params.get("psf_zerns", {}))
        
        self.pupil = pupil_mag * np.exp(1j*pupil_phase)
        self.pupil = self.pupil[pupil_padding[0]:pupil_padding[1], pupil_padding[2]:pupil_padding[3]]
        self.pupil_suppl = {"radial_distance": (R*(R<=1))[pupil_padding[0]:pupil_padding[1], pupil_padding[2]:pupil_padding[3]],
                           "azimuthal_angle": np.arctan2(US, VS)[pupil_padding[0]:pupil_padding[1], pupil_padding[2]:pupil_padding[3]]}
        
        shifted_pupil_phase = np.tile(pupil_phase, (shifts.shape[0], 1, 1))
        shifted_pupil_phase = shifted_pupil_phase - 2 * np.pi * (self.KX[None,...] * shifts[:,0,None,None])
        shifted_pupil_phase = shifted_pupil_phase - 2 * np.pi * (self.KY[None,...] * shifts[:,1,None,None])        
        shifted_pupil_phase = shifted_pupil_phase + np.sqrt(1-np.minimum(R, 1)**2) * shifts[:,2,None,None]        
        
        shifted_pupils = pupil_mag[None,...]*np.exp(1j*shifted_pupil_phase)
        
        psfs = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(shifted_pupils)))
        psfs = psfs[:, pupil_padding[0]:pupil_padding[1], pupil_padding[2]:pupil_padding[3]]
        psfs = np.abs(psfs)**2
        
        ref_psf = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(np.pad(self.pupil, ((pupil_padding[0], -pupil_padding[1]), (pupil_padding[2], -pupil_padding[3]))))))
        ref_psf = ref_psf[pupil_padding[0]:pupil_padding[1], pupil_padding[2]:pupil_padding[3]]
        ref_psf = np.abs(ref_psf)**2
        
        psfs /= ref_psf.max()
        
        return psfs


class FileWrapperDataset(Dataset):
    def __init__(self, file_path, file_loader, slices=(slice(None),), stack_to_volume=False, cache=True):
        self.file = file_loader(file_path, slices=slices, stack_to_volume=stack_to_volume, cache=cache)
        print(", ".join(["{}: {}".format(key, val) for key, val in {"filepath":self.file.file_path, "frames":len(self.file), "image shape":self.file[0].shape}.items()]))
    
    def __len__(self):
        return len(self.file)
    
    def __getitem__(self, key):
        return self.file[key], {'id': key}


def inspect_images(dataset, indices=None):
    if indices is None:
        indices = np.random.choice(len(dataset), min(8, len(dataset)), replace=False)
    images, labels = zip(*[dataset[i] for i in indices])
    
    tiled_images, n_col, n_row  = util.tile_images(util.reduce_images_dim(np.stack(images, axis=0)), full_output=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(4*n_col, 3*n_row*2))
    im = axes[0].imshow(tiled_images)
    plt.colorbar(im, ax=axes[0])
    im = axes[1].imshow(np.log(tiled_images))
    plt.colorbar(im, ax=axes[1])
    
    for i, id in enumerate(indices):
        label = "{}:\t".format(id)
        for key, val in labels[i].items():
            label += " [{} =".format(key)
            for datum in np.atleast_1d(val.squeeze()):
                label += " {:.3f},".format(datum)
            label += "],"
        print(label)
        for j in range(2):
            axes[j].text(i%n_col / n_col, i//n_col / n_row,
                         # label,
                         id,
                         bbox={'facecolor':'white', 'alpha':1},
                         ha='left', va='bottom',
                         fontsize='medium',
                         transform=axes[j].transAxes)
            
    if hasattr(dataset, 'params'):
        fig, axes = plt.subplots(1, len(dataset.params), figsize=(4*len(dataset.params), 3))
        for i, (key, val) in enumerate(dataset.params.items()):
            axes[i].hist(val.flatten(), bins=20)
            axes[i].set_xlabel(key)
    
    if hasattr(dataset, 'pupil'):
        fig, axes = plt.subplots(1, 3, figsize=(4*2 + 8, 3), gridspec_kw={'width_ratios': [1,1,3]})
        pupil_magnitude = np.abs(dataset.pupil)
        pupil_magnitude_colored, norm, cmap = util.color_images(pupil_magnitude, full_output=True)
        im = axes[0].imshow(pupil_magnitude_colored)
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[0])
        axes[0].set_title('pupil mag')
        
        pupil_phase = restoration.unwrap_phase(np.ma.array(np.angle(dataset.pupil), mask=np.abs(dataset.pupil)<=0))
        pupil_phase_colored, norm, cmap = util.color_images(pupil_phase, vsym=True, full_output=True)
        im = axes[1].imshow(pupil_phase_colored)
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[1])
        axes[1].set_title('pupil phase')
        
        zernike_coeffs = zernike.fit_zernike_from_pupil(dataset.pupil, 16, dataset.pupil_suppl["radial_distance"], dataset.pupil_suppl["azimuthal_angle"])        
        zernike.plot_zernike_coeffs(axes[2], zernike_coeffs)
        
        fig.tight_layout()