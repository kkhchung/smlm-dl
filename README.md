# smlm-dl

This package was written to experiment with a high-level architecture to perform **unsupervised learning** on **image-based data**. It shares similarity with an encoder-decoder model, with the main exceptions that the decoder is based on a physical model (but may also containing trainable parameters), and the 'encoder' itself is not restricted to  a classic encoder network.
Originally written for localization of single-molecule image data, but the general architecture has proven adaptable to perform other image analyses such as deconvolution and denoising.

The model/code is designed to be highly modularized:

Module|Description
---|---
`fit`|Wrapper classes to link all the components. At least contain an encoder, a mapper, and a renderer.|
`encoder`|Bulk of the network. Designed to extract 'useful' information from images. E.g. CNN encoder to extract features or U-Net to transform images.
`mapper`|Converts the 'unlabeled' output from an `encoder` to meaningful parameters that the `renderer` can interpret. May also contain trainable parameters.
`renderer`|Physical model to render images. May also contain trainable parameters.

* The input and final predicted output are always images and the loss is based on their comparison and therefore self-supervised and does not require labels.
* The loss is expected to follow camera noise models (mixed Poisson-Gaussian) and can be accounted for in custom loss functions.

## Image analysis capabilities

* Fit single PSFs using 2/3D-Gaussian function to determine PSF position, intensity and width
* Fit single aberated PSFs using Fourier optics model to determine PSF properties and extract the pupil-plane phase
* Fit multiple PSFs
* (Blind) Deconvolution
* Denoising of pixel-based noise

## Features

* Point spread function (PSF) simulator (2/3-D Gaussian and Fourier optics model)
* PSF, convolution, and template-based renderer
* FFT- and spline-based 2/3-D interpolation
* All models are CUDA-compatible
* Uses Tensorboard for logging and profiling

## Installation

Requires installation as a package.
For example, with pip, where `path` is the directory where smlm-dl is installed:
```
pip install -e <path>
```

## Configuration

Persistent settings are written to `config.ini`.

Setting | Description
--- | --- |
`[LOG_PATH] \ run` | Location to save TensorBoard logs and checkpoints.
`[TEST_DATASET_PATH] \ ...` | Locations where datasets are saved.
 

## Special folders

Folder | Description
--- | ---
[examples](/smlm_dl/examples) | Basic examples. Not fully trained.
[tests](/smlm_dl/tests) | Jupyter notebooks for testing code. More advanced examples can also be found here.


## Datasets

Some examples requires public datasets that are not provided here.

Author | Link
--- | ---
EPFL, Biomedical Imaging Group | http://bigwww.epfl.ch/deconvolution/
Hagen, et al. 2021 | https://doi.org/10.1093/gigascience/giab032 ,  http://dx.doi.org/10.5524/100888
