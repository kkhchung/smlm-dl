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