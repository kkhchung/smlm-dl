import numpy as np

import torch
from torch import nn

from matplotlib.pyplot import *

class Spline2D(nn.Module):
    def __init__(self, image, k=3, out_size=None, fit_image=True):
        super().__init__()
        self.k = k
        self.template_size = image.shape
        if out_size is None:
            self.out_size = image.shape
        else:
            self.out_size = out_size
        
        coeffs = torch.tensor(image, dtype=torch.float)
        padding = 4
        coeffs = torch.nn.functional.pad(coeffs, (padding,)*4)
        self.coeffs = nn.Parameter(coeffs)
        self.offset = padding - (k+1)//2
        self.centering_offset = [(self.out_size[0] - self.template_size[0])//2,
                                 (self.out_size[1] - self.template_size[1])//2,]
        if (any(val<0 for val in self.centering_offset)):
            raise Exception("Out_size must be larger than size of image. Limitation of using torch.scatter.")
        
        if fit_image:
            self._fit_image(image)
        
        index_a = torch.arange(self.template_size[1]+1).tile(self.template_size[0]+1, 1)
        index_b = torch.arange(self.out_size[0]+2)[:,None].tile(1, self.out_size[1]+2)
        self.register_buffer('index_a', index_a)
        self.register_buffer('index_b', index_b)
        
    def forward(self, x, raw=False):
        if x is None:
            x = dict()
            x['x'] = (torch.randn((1024, 1, 1, 1)) - 0.5) * 10
            x['y'] = (torch.randn((1024, 1, 1, 1)) - 0.5) * 10
        shifts = x
        shift_pixel = {key: torch.floor(val/1).type(torch.int) for key, val in shifts.items()}
        shifts_subpixel = {key: torch.remainder(val, 1) for key, val in shifts.items()}
        batch_size = list(x.values())[0].shape[0]
        
        template = self._render_template(shifts_subpixel)
        if raw:
            return template
        
        x_a = torch.zeros((batch_size, 1, self.out_size[0]+2, self.out_size[1]+2))
        
        index = self.index_a.tile(batch_size, 1, 1, 1) + shift_pixel['y'] + self.centering_offset[1] + 1
        index = torch.clamp(index, 0, x_a.shape[3]-1)
        x_a.scatter_(3, index, template)
        
        x_b = torch.zeros_like(x_a)
        
        index = self.index_b.tile(batch_size, 1, 1, 1) + shift_pixel['x'] + self.centering_offset[0] + 1
        index = torch.clamp(index, 0, x_b.shape[2]-1)
        x_b.scatter_(2, index, x_a)
        
        x = x_b[:,:,1:-1,1:-1]
        
        return x
    
    def _render_template(self, shifts_subpixel):
        bases = {key: calculate_bspline_basis(val, self.k) for key, val in shifts_subpixel.items()}
        x = 0
        for i, cx in enumerate(bases['x']):
            for j, cy in enumerate(bases['y']):
                coeffs = self.coeffs[i+self.offset:i+self.offset+self.template_size[0]+1,
                                     j+self.offset:j+self.offset+self.template_size[1]+1]
                x = x + cx * cy * coeffs
        return x
    
    def render(self, scale=1):
        x = torch.arange(0, 1, 1/scale)
        y = torch.arange(0, 1, 1/scale)
        xs, ys = torch.meshgrid(x, y, indexing='ij')

        x = dict()
        x['x'] = xs.flatten()[:,None,None,None]
        x['y'] = ys.flatten()[:,None,None,None]
        
        interpolated_images = self._render_template(x)
        image_highres = np.empty(((self.template_size[0]+1)*scale, (self.template_size[1]+1)*scale))
        
        for i, img in enumerate(interpolated_images):
            c = i // scale
            r = i % scale
            
            image_highres[scale-c-1::scale, scale-r-1::scale] = interpolated_images[i,0].detach()
        
        image_highres = image_highres[scale//2:-((scale+1)//2), scale//2:-((scale+1)//2)]
        return image_highres
    
    def _fit_image(self, image_groundtruth, maxiter=200, tol=1e-9):
        image_groundtruth = torch.as_tensor(image_groundtruth).type(torch.float32)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=10)
        
        x = {'x':torch.zeros((1,1,1,1)),
             'y':torch.zeros((1,1,1,1))}
        
        print("Fitting...")
        loss_log = list()
        for i in range(maxiter):
            optimizer.zero_grad()
            pred = self(x, raw=True)
            loss = loss_function(pred[0,0,:-1,:-1], image_groundtruth)
            loss_log.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()
            if (loss < tol):
                print("Early termination, loss tol < {} reached".format(tol))
                break
            
        # plot(loss_log)
        print("Final loss: {}".format(loss_log[-1]))
        return loss_log


def B(i, p, k):
    if k == 0:
        return 1 if i == 0 else 0
    c1 = (p + i) / k * B(i, p, k-1)
    c2 = (k - i + 1 - p) / k * B(i-1, p, k-1)
    return c1 + c2


def calculate_bspline_basis(p, k, fast=True):
    if fast and k>=1 and k<=3:
        if k == 1:
            res = [p, 1-p]
        elif k == 2:
            res = [p**2/2,
                   -p**2 + p + 1/2,
                   (1-p)**2/2]
        elif k == 3:
            res = [p**3/6,
                   -p**3/2 + p**2/2 + p/2 + 1/6,
                   p**3/2 - p**2 + 2/3,
                   (1-p)**3/6]
    else:
        res = list()
        for i in range(k+1):
            res.append(B(i, p, k))
    return res

def test_calculate_bspline_basis_fast_mode(iter=1000):
    ks = np.random.randint(1, 4, size=iter)
    allclose = True
    for i in range(iter):
        p = np.random.rand(9)
        res_fast = calculate_bspline_basis(p, ks[i], True)
        res_slow = calculate_bspline_basis(p, ks[i], False)
        same = all([np.allclose(a, b) for a, b in zip(res_fast, res_slow)])
        if same is False:
            allclose = False
            print("FAILED\tk={},\nfast: {}\nslow: {}".format(ks[i], res_fast, res_slow))
    if allclose is True:
        print("PASSED all {} runs.".format(iter))
