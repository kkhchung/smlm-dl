import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import torch
from torch import nn

import util


class BaseModel(nn.Module):
    def call_auto(self, x, y=None):
        # 'smart' call that workers for both id / image type encoder
        if y is None:
            return nn.Module.__call__(self, x)
        else:
            if self.image_input:
                pred = nn.Module.__call__(self, x)
            else:
                pred = nn.Module.__call__(self, y["id"])
            return pred


class ParameterModule(nn.Module):
    """
    Hack
    Wrapper class for nn.Parameter so that it will show up with pytorch.summary
    """
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        self.parameter = nn.Parameter(*args, **kwargs)
    
    def forward(self, x=None):
        return self.parameter


class ViewModule(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.view(-1, *self.shape)


class FitParameter(object):
    activation = None
    offset = None
    scaling = None
    per_psf = None
    
    def __init__(self, activation, offset, scaling, default, per_psf):
        self.activation = activation
        self.offset = offset
        self.scaling = scaling
        self.default = default
        self.per_psf = per_psf
        
    def copy(self):
        return FitParameter(self.activation, self.offset, self.scaling, self.default, self.per_psf)
    
    def __repr__(self):
        text = "<"
        text += "act: {}, ".format(self.activation.__class__.__name__)
        text += "offset: {}, ".format(self.offset)
        text += "scaling: {}, ".format(self.scaling)
        text += "default: {}, ".format(self.default)
        text += "per psf: {}, ".format(self.per_psf)
        text += ">"
        return text


def check_model(model, dataloader=None):
    is_training = model.training
    model.train(False)
    
    if not dataloader is None:
        features, labels = next(iter(dataloader))
        pred = model.call_auto(features, labels)
        pred = pred.detach().numpy()
        features = features.detach().numpy()
        
        print("input shape: {}, output_shape: {}".format(features.shape, pred.shape))
        
        pred_is_image = pred.shape[2]>1 and pred.shape[3]>1
        
        n_col = 8
        n_images = 8
        
        features_images = util.reduce_images_dim(features[:n_images])
        features_tiled, n_col, n_row = util.tile_images(features_images, n_col, full_output=True)
        
        fig, axes = plt.subplots(3 if pred_is_image else 1, 1,
                                 figsize=(4*n_col, 3*n_row*(3 if pred_is_image else 1)),
                                 squeeze=False)
        im = axes[0,0].imshow(features_tiled)
        plt.colorbar(im, ax=axes[0,0])
        axes[0,0].set_title("data")
        
        if pred_is_image:
            pred_images = util.reduce_images_dim(pred[:n_images])
            pred_tiled, n_col, n_row = util.tile_images(pred_images, n_col, full_output=True)
            im = axes[1,0].imshow(pred_tiled)
            plt.colorbar(im, ax=axes[1,0])
            axes[1,0].set_title("predicted")
        
            diff = pred_tiled-features_tiled
            diff_max = max(abs(diff.min()), diff.max())
            im = axes[2,0].imshow(diff, cmap="seismic", vmin=-diff_max, vmax=diff_max)
            plt.colorbar(im, ax=axes[2,0])
            axes[2,0].set_title("diff")
    
    if hasattr(model, 'get_suppl'):
        suppl_dict = model.get_suppl(colored=True)
        if 'images' in suppl_dict:
            suppl_images = suppl_dict['images']
            fig, axes = plt.subplots(1, len(suppl_images), figsize=(4*len(suppl_images), 3), squeeze=False)
            for i, (key, (img, norm, cmap)) in enumerate(suppl_images.items()):
                im = axes[0, i].imshow(img)
                plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[0, i])
                axes[0, i].set_title(key)
        if 'plots' in suppl_dict:
            suppl_plots = suppl_dict['plots']
            for key, val in suppl_plots.items():
                fig, ax = plt.subplots(1, 1, **val['kwargs'])
                val['plot'](ax)
                
    if hasattr(model, 'render_example_images'):
        example_images = model.render_example_images(8)
        example_images = util.reduce_images_dim(example_images)
        fig, axes = plt.subplots(1, len(example_images), figsize=(4*len(example_images), 3), squeeze=False)
        for i, img in enumerate(example_images):
            im = axes[0, i].imshow(img)
            plt.colorbar(im, ax=axes[0, i])
            axes[0, i].set_title("E.g. {}".format(i))
            
        fig, axes = plt.subplots(1, len(example_images), figsize=(4*len(example_images), 3), squeeze=False)
        for i, img in enumerate(example_images):
            im = axes[0, i].imshow(np.log10(img))
            plt.colorbar(im, ax=axes[0, i])
            axes[0, i].set_title("E.g. {}".format(i))
            
    model.train(is_training)