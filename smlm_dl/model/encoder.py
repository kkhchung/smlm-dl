import functools
import numpy as np
import torch
from torch import nn

from . import base


class BaseEncoderModel(base.BaseModel):
    def __init__(self, last_out_channels=2, **kwargs):
        super().__init__()
        self.last_out_channels = last_out_channels
        self.build_model(**kwargs)
        
    def build_model(self, **kwargs):
        raise NotImplementedError()


class IdEncoderModel(BaseEncoderModel):
    image_input = False
    
    def __init__(self, num_img, last_out_channels=2, out_img_shape=(1,1,), init_weights=None):
        self.out_img_shape = out_img_shape
        super().__init__(last_out_channels=last_out_channels, **{"num_img":num_img, "init_weights":init_weights})
    
    def build_model(self, num_img, init_weights=None):
        self.one_hot = functools.partial(torch.nn.functional.one_hot, num_classes=num_img)
        out_shape = [self.last_out_channels,] + list(self.out_img_shape)
        self.encoders = nn.ModuleDict()
        self.encoders["scale"] = nn.Linear(num_img, np.prod(out_shape), bias=False)
        self.encoders["view"] = base.ViewModule(out_shape)
        if not init_weights is None:
            with torch.no_grad():
                self.encoders["scale"].weight.view(-1).copy_(torch.as_tensor(init_weights).view(-1))
    
    def forward(self, x):
        x = x.to(torch.long)
        x = self.one_hot(x)
        x = x.to(torch.float)
        for key, val in self.encoders.items():
            x = val(x)
        return x
    
    
class ImageEncoderModel(BaseEncoderModel):
    image_input = True
    
    def __init__(self, img_size=(32,32), in_channels=1, last_out_channels=2, **kwargs):
        if not img_size is None and (img_size[0] % 2 != 0 or img_size[1] % 2 !=0):
            raise Exception("Image input size needs to be multiples of two.")
        self.img_size = img_size
        self.in_channels = in_channels
        super().__init__(last_out_channels=last_out_channels, **kwargs)
        

class ConvImageEncoderModel(ImageEncoderModel):
    def __init__(self, img_size=(32,32), depth=3, in_channels=1, first_layer_out_channels=16, last_out_channels=2, skip_channels=0,):
        if 2**depth > img_size[0] or 2**depth > img_size[1]:
            raise Exception("Model too deep for this image size (depth = {}, image size needs to be at least ({}, {}))".format(depth, 2**depth, 2**depth))
        
        super().__init__(img_size=img_size, in_channels=in_channels,
                                  last_out_channels=last_out_channels,
                                  **{"depth":depth, "first_layer_out_channels":first_layer_out_channels, "skip_channels":skip_channels})
        
    def build_model(self, depth, first_layer_out_channels, skip_channels):        
        self.encoders = nn.ModuleDict()
        self.skips = nn.ModuleDict()
        for i in range(depth):
            in_channels = self.in_channels if i == 0 else 2**(i-1) * first_layer_out_channels
            out_channels = 2**i * first_layer_out_channels            
            if skip_channels > 0:
                in_shape = (int(self.img_size[0] * 0.5**i), int(self.img_size[1] * 0.5**i))
                self.skips["skip_conv_layer{}".format(i)] = self._skip_block(in_channels, skip_channels, in_shape)
            self.encoders["conv_layer{}".format(i)] = self._encode_block(in_channels, out_channels)
            
        self.neck = nn.ModuleDict()
        self.neck["conv_layer_0"] = self._neck_block(int(2**(depth-1) * first_layer_out_channels),
                                             (2**(depth) * first_layer_out_channels))
        self.neck["conv_layer_1"] = self._encode_final_block(2**(depth) * first_layer_out_channels,
                                                             2**(depth-1) * first_layer_out_channels,
                                                            tuple(int(size*0.5**depth) for size in self.img_size))
        
        self.decoders = nn.ModuleDict()
        self.decoders["dense_layer_0"] = nn.Conv2d(2**(depth-1) * first_layer_out_channels + skip_channels*depth,
                                                   self.last_out_channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        skips = dict()
        
        if len(self.skips) > 0:        
            for i, ((encoder_key, encoder_module), (skip_key, skip_module)) in enumerate(zip(self.encoders.items(), self.skips.items())):
                skips[skip_key] = skip_module(x)
                x = encoder_module(x)
        else:
            for i, (key, module) in enumerate(self.encoders.items()):
                x = module(x)
        
        for i, (key, module) in enumerate(self.neck.items()):
            x = module(x)

        x = torch.cat([x] + list(skips.values()), dim=1)
        for i, (key, module) in enumerate(self.decoders.items()):
            x = module(x)
            
        return x
    
    def _skip_block(self, in_channels, out_channels, in_shape):
        return nn.Sequential(            
            nn.GroupNorm(in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=in_shape, padding=0),
            nn.GELU(),
            nn.GroupNorm(out_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Dropout2d(0.5),
        )
    
    def _encode_block(self, in_channels, out_channels):
        return nn.Sequential(
            # nn.GroupNorm(in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            # nn.GroupNorm(out_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5),
        )
    
    def _neck_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.GroupNorm(in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(out_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout2d(0.5),
            nn.GELU(),
        )
    
    def _encode_final_block(self, in_channels, out_channels, image_shape):
        return nn.Sequential(
            nn.GroupNorm(in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=image_shape, padding=0),
            nn.GELU(),
        )
    
    
class UnetEncoderModel(ImageEncoderModel):
    def __init__(self, img_size=(32,32), depth=3, in_channels=1, first_layer_out_channels=16, last_out_channels=2, ):
        if 2**depth > img_size[0] or 2**depth > img_size[1]:
            raise Exception("Model too deep for this image size (depth = {}, image size needs to be at least ({}, {}))".format(depth, 2**depth, 2**depth))
        
        super().__init__(img_size=img_size, in_channels=in_channels,
                                  last_out_channels=last_out_channels,
                                  **{"depth":depth, "first_layer_out_channels":first_layer_out_channels, })
        
    def build_model(self, depth, first_layer_out_channels, ):
        self.encoders = nn.ModuleDict()
        for i in range(depth):
            in_channels = self.in_channels if i == 0 else 2**(i-1) * first_layer_out_channels
            out_channels = 2**i * first_layer_out_channels

            self.encoders["conv_layer{}".format(i)] = self._conv_block(in_channels, out_channels)
            self.encoders["pool_layer{}".format(i)] = self._pool_block(out_channels, out_channels)
            
        self.neck = self._conv_block(int(2**(depth-1) * first_layer_out_channels),
                                     (2**(depth) * first_layer_out_channels))
        
        self.decoders = nn.ModuleDict()
        for i in range(depth):
            in_channels = 2**(depth - i) * first_layer_out_channels
            out_channels = 2**(depth - i - 1) * first_layer_out_channels
            self.decoders["up_conv_layer{}".format(i)] = self._up_conv_block(in_channels, out_channels)
            self.decoders["conv_layer{}".format(i)] = self._conv_block(in_channels, out_channels)
        
        self.final_decoder = self._final_decoder(first_layer_out_channels, self.last_out_channels)
        
    def forward(self, x):
        skips = list()
        
        for i, (key, module) in enumerate(self.encoders.items()):
            x = module(x)
            if i % 2 == 0: # is conv_layer
                skips.insert(0, x)
        
        x = self.neck(x)
        
        for i, (key, module) in enumerate(self.decoders.items()):
            if i % 2 == 1: # is conv_layer
                x = torch.cat([skips[i//2], x], dim=1)
            x = module(x)
            
        x = self.final_decoder(x)
            
        return x
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            # nn.GroupNorm(in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.GroupNorm(out_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Dropout2d(0.5),
        )
    
    def _pool_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2),
        )
    
    def _up_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        )
    
    def _final_decoder(self, in_channels, out_channels):
        return nn.Sequential(
            # nn.GroupNorm(in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.ReLU(),
        )