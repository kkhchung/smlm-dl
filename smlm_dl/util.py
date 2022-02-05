import numpy as np
import matplotlib

def color_images(img, cmap=None, vmin=None, vmax=None, vsym=False, full_output=False):
    # convert N, 1, H, W data to N, C, H, W
    # or H, W to H, W, C
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
    else:
        return colored_img

def tile_images(img, n_col=8, full_output=False):
    # convert N, C, H, W data to 1, C, row*H, col*W
    # or N, H, W to row*H, col*W
    img_count = img.shape[0]
    n_col = min(n_col, img_count)
    n_row = np.ceil(img_count / n_col).astype(int)
    remainder = n_col * n_row - img_count
    img_tiled = np.pad(img, ((0, remainder),) + ((0,0),)*(len(img.shape)-1))
    img_tiled = img_tiled.reshape((n_row, n_col, -1, img.shape[-2], img.shape[-1]))
    img_tiled = np.moveaxis(img_tiled, (0,1), (-4,-2))
    img_tiled = img_tiled.reshape((1, -1, n_row*img.shape[-2], n_col*img.shape[-1]))
    if img.ndim <=3:
        img_tiled = img_tiled[0,0]
    
    if full_output:
        return img_tiled, n_col, n_row
    else:
        return img_tiled


def reduce_images_dim(images, ch_func="mean", out_dim=3, dim_func="max"):
    # expect N, C, H, W, ...
    if ch_func == "mean":
        images = images.mean(1)
    elif ch_func == "max":
        images = images.max(1)
    elif ch_func == "skip":
        out_dim += 1
    else:
        raise Exception("ch_func not recognized")
    
    extra_dim = images.ndim - out_dim
    extra_dim = tuple(np.arange(-extra_dim, 0, 1))
    if dim_func == "mean":
        images = images.mean(axis=extra_dim)
    elif dim_func == "max":
        images = images.max(axis=extra_dim)
    elif dim_func == "middle":
        for dim in extra_dim:
            images = images[...,images.shape[-1]//2]
    else:
        raise Exception("dim_func not recognized")
    
    return images


def calculate_padding(init_size, final_size, flat=True):
    padding = [(sizes[1] - sizes[0])//2 for sizes in zip(init_size, final_size)]
    padding = list()
    for size in zip(init_size, final_size):
        diff = size[1] - size[0]
        if diff < 0:
            raise Exception("final_size needs to be equal or larger than init_size in all dimensions")
        if not diff % 2 == 0:
            raise Exception("odd amount of padding not implemented")
        padding.append(diff//2)
    
    if flat:
        padding = np.repeat(padding, 2)
    else:
        padding = [(pad, pad) for pad in padding]
    
    return padding


def calculate_slicing(init_size, final_size):
    padding = list()
    for size in zip(init_size, final_size):
        diff = size[0] - size[1]
        if diff < 0:
            raise Exception("final_size needs to be equal or smaller than init_size in all dimensions")
        if not diff % 2 == 0:
            raise Exception("odd amount of padding not implemented")
        padding.append(diff//2)
    padding_slice = tuple([slice(None) if pad==0 else slice(pad, -pad) for pad in padding])
    
    return padding_slice