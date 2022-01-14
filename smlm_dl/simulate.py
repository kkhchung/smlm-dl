import numpy as np
import matplotlib

def simulate_image_gauss2d(width, height, num=1, sig=2, full_output=False):
    x, y = np.linspace(0, width-1, width), np.linspace(0, height-1, height)
    X, Y = np.meshgrid(x, y, indexing='ij')
    points_x = np.random.random(num) * width
    points_y = np.random.random(num) * height
    
    image = np.zeros((width, height))
    for (pt_x, pt_y) in zip(points_x, points_y):
        image  += np.exp(-((X-pt_x)**2/(2*sig**2)+(Y-pt_y)**2/(2*sig**2)))
        
    if full_output:
        return image, (x, y)
    else:
        return image
    
def simulate_centered_2d_gaussian(width, height, sig=2):
    image = simulate_centered_3d_gaussian(width, height, 1, sig)[:,:,0]
    
    return image

def simulate_centered_3d_gaussian(width, height, depth, sig_xy=2, sig_z=6):
    sig_x = sig_xy
    sig_y = sig_xy
    xs = np.linspace(-1, 1, width) * width * 0.5
    ys = np.linspace(-1, 1, height) * height * 0.5
    zs = np.linspace(-1, 1, depth) * depth * 0.5
    xs, ys, zs = np.meshgrid(xs, ys, zs, indexing='ij')
    volume = np.exp(-(0.5*xs**2/sig_x**2 + 0.5*ys**2/sig_y**2 + 0.5*zs**2/sig_z**2))
    
    return volume