import numpy as np
import matplotlib

def color_images(img, cmap=None, vmin=None, vmax=None, vsym=False, full_output=False):
    # convert N, 1, H, W data to N, C, H, W where C s
    if vmin is None:
        vmin = np.nanmin(img)
    if vmax is None:
        vmax = np.nanmax(img)
    if vsym == True:
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
    normalizer = matplotlib.colors.Normalize(vmin, vmax)
    if isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap)
    elif cmap is None:
        if vsym == False:
            cmap = matplotlib.cm.get_cmap('viridis')
        else:
            cmap = matplotlib.cm.get_cmap('seismic')        
    colored_img =  cmap(normalizer(img))
    if len(colored_img.shape) > 3:
        colored_img = np.swapaxes(colored_img, 1, -1)[:,:3,:,:,0]
    
    if full_output:
        return colored_img, normalizer, cmap
    return colored_img

def tile_images(img, n_col=8):
    img_count = img.shape[0]
    n_row = np.ceil(img_count / n_col).astype(int)
    remainder = n_col - (img_count % n_row)
    img_tiled = np.pad(img, ((0, remainder),) + ((0,0),)*(len(img.shape)-1))    
    img_tiled = np.vstack([np.hstack(img_tiled[r*n_col:(r+1)*n_col]) for r in range(n_row)])
    return img_tiled, n_col, n_row