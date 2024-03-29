import numpy as np
import math
from skimage import restoration
import torch

DEBUG = False

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

def calculate_pupil_from_zernike(j, radial_distance, azimuthal_angle):  
    m, n = decode_ansi_index(j)
        
    z = calculate_radial_polynomial(np.abs(m), n, radial_distance)
    
    if m >=  0:
        z *= np.cos(m * azimuthal_angle)
    else:
        z *= np.sin(np.abs(m) * azimuthal_angle)
    
    return z

def calculate_pupil_phase(radial_distance, azimuthal_angle, zernikes):
    if torch.is_tensor(radial_distance):
        pupil_phase = torch.zeros(radial_distance.shape)
    else:
        pupil_phase = np.zeros(radial_distance.shape)
    for key, val in zernikes.items():
        pupil_phase += val * calculate_pupil_from_zernike(key, radial_distance, azimuthal_angle) * (radial_distance <= 1)
    return pupil_phase

def fit_zernike_from_pupil(pupil, max_j, radial_distance, azimuthal_angle):
    mask = np.abs(pupil) <= 0
    
    # unwrapping can cause error in zernike mode zero
    pupil_phase = restoration.unwrap_phase(np.ma.array(np.angle(pupil), mask=mask))
    radial_distance = np.ma.array(radial_distance, mask=mask)
    azimuthal_angle = np.ma.array(azimuthal_angle, mask=mask)
    
    if DEBUG:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(4*3, 3))
        axes[0].imshow(pupil_phase)
        axes[1].imshow(radial_distance)
        axes[2].imshow(azimuthal_angle)
    
    zernike_basis = list()    
    for j in range(max_j):
        zernike_basis.append(calculate_pupil_from_zernike(j, radial_distance, azimuthal_angle))
    
    zernike_coeff = {}
    for j, basis in enumerate(zernike_basis):
        res = np.linalg.lstsq(basis.compressed().reshape(-1, 1), pupil_phase.compressed().reshape(-1), rcond=None)[0]
        zernike_coeff[j] = res[0]

    if DEBUG:
        for key, val in zernike_coeff.items():
            print("{}: {:.3f}".format(key, val))
    
    return zernike_coeff

def plot_zernike_coeffs(ax, zernike_coeffs):
    ax.axhline(0, c='black')
    ax.bar(zernike_coeffs.keys(), zernike_coeffs.values())
    for key, val in zernike_coeffs.items():
        ax.annotate("{:.2f}".format(val), (key, val), ha='center', va='bottom' if (np.sign(val)>=0) else 'top')
    ticks = list(zernike_coeffs.keys())
    ax.set_xlim(0-0.5, max(ticks)+0.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["{}".format(t) for t in ticks], fontsize='small')
    ax.set_xlabel("zernike mode")
    y_lim = max(-1*min(zernike_coeffs.values()), max(zernike_coeffs.values())) * 1.3
    ax.set_ylim(-y_lim, y_lim)
    
def compensate_tip_tilt_flatten(pupil_phase, mask):
    # might be faster, flattens down one dimension
    pupil_phase_masked = restoration.unwrap_phase(np.ma.array(pupil_phase, mask=mask))
    correction_phase = np.zeros_like(pupil_phase_masked)
    
    for i in range(2):
        j = 1-i
        y = pupil_phase_masked.mean(axis=j)
        x = np.ma.array(np.arange(y.shape[0])-0.5*(y.shape[0]-1), mask=np.ma.getmask(y))
        
        b = np.linalg.lstsq(x.compressed().reshape(-1,1), y.compressed(), rcond=None)[0]        
        pred = x * b
        
        if i == 0:
            correction_phase += pred[:,None]
        elif i ==1:
            correction_phase += pred[None,:]
            
        if DEBUG:
            print(b)
            import matplotlib.pyplot as plt
            fig, ax= plt.subplots(1, 1)
            ax.scatter(x, y)
            ax.scatter(x, pred)            
    
    corrected_phase = pupil_phase_masked - correction_phase
    
    if DEBUG:        
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12,3))
        im = axes[0].imshow(pupil_phase_masked)
        plt.colorbar(im, ax=axes[0])
        im = axes[1].imshow(correction_phase)
        plt.colorbar(im, ax=axes[1])
        im = axes[2].imshow(corrected_phase)
        plt.colorbar(im, ax=axes[2])
    
    return corrected_phase

def compensate_tip_tilt(pupil_phase, mask):
    pupil_phase_masked = restoration.unwrap_phase(np.ma.array(pupil_phase, mask=mask))
    correction_phase = np.zeros_like(pupil_phase_masked)
    xs = np.arange(pupil_phase_masked.shape[0])-0.5*(pupil_phase_masked.shape[0]-1)
    ys = np.arange(pupil_phase_masked.shape[1])-0.5*(pupil_phase_masked.shape[1]-1)
    XS, YS = np.meshgrid(xs, ys, indexing='ij')
    XS = np.ma.array(XS, mask=mask)
    YS = np.ma.array(YS, mask=mask)
    coord = [XS, YS]
    
    for i in range(2):
        j = 1-i
        y = pupil_phase_masked
        x = coord[i]
        
        b = np.linalg.lstsq(x.compressed().reshape(-1,1), y.compressed(), rcond=None)[0]        
        pred = x * b
        
        correction_phase += pred
            
        if DEBUG:
            print(b)
            import matplotlib.pyplot as plt
            fig, ax= plt.subplots(1, 1)
            ax.scatter(x, y)
            ax.scatter(x, pred)            
    
    corrected_phase = pupil_phase_masked - correction_phase
    
    if DEBUG:        
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12,3))
        im = axes[0].imshow(pupil_phase_masked)
        plt.colorbar(im, ax=axes[0])
        im = axes[1].imshow(correction_phase)
        plt.colorbar(im, ax=axes[1])
        im = axes[2].imshow(corrected_phase)
        plt.colorbar(im, ax=axes[2])
    
    return corrected_phase
                
class PupilPhase(object):
    def __init__(self, U, V):
        self.U = U
        self.V = V
        self.radial_distance = np.sqrt(U**2 + V**2)
        self.mask = self.radial_distance > 1
        self.azimuthal_angle = np.arctan2(V, U)
        
        self.tilt = np.ma.array(self.calculate_phase({1:1}), mask=self.mask)
        self.tip = np.ma.array(self.calculate_phase({2:1}), mask=self.mask)
        self.defocus = np.ma.array(self.calculate_phase({4:1}), mask=self.mask)
        
    def calculate_phase(self, zernikes):
        return calculate_pupil_phase(self.radial_distance, self.azimuthal_angle, zernikes)
        
    def remove_tilt_tip_defocus(self, pupil_phase):
        pupil_phase_masked = restoration.unwrap_phase(np.ma.array(pupil_phase, mask=self.mask))
        correction_phase = np.zeros_like(pupil_phase_masked)
        for coeffs in [self.defocus, self.tip, self.tilt]:
            b = np.linalg.lstsq(coeffs.compressed().reshape(-1,1), pupil_phase_masked.compressed(), rcond=None)[0]
            pred = coeffs * b
            correction_phase += pred
        return correction_phase