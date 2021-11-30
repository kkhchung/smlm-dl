import numpy as np
import math

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
    pupil_phase = np.zeros(radial_distance.shape)
    for key, val in zernikes.items():
        pupil_phase += val * calculate_pupil_from_zernike(key, radial_distance, azimuthal_angle)
    return pupil_phase