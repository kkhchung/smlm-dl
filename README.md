# smlm-dl



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
*examples* | Basic examples.
*tests* | Jupyter notebooks for testing code. More advanced examples can be found here.
*experiments* | Experimental code. Maybe non-functional.

## Datasets
Some examples requires public datasets that are not provided here.

Author | Link
--- | ---
EPFL, Biomedical Imaging Group | http://bigwww.epfl.ch/deconvolution/
Hagen, et al. | https://doi.org/10.1093/gigascience/giab032 http://dx.doi.org/10.5524/100888
