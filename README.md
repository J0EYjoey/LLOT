# LLOT: application of Laplacian Linear Optimal Transport in spatial transcriptome reconstruction

LLOT is a computational method to integrate a spatial transcriptomic
dataset and a reference scRNA-seq dataset to perform the three tasks: spatial expression prediction, cell location inference and cell-type deconvolution.

## Dependency

LLOT requires packages listed as follows:
- numpy, pot, scipy, scikit-learn, pandas
- LLOT related functions are available in [LLOT_util.py](https://github.com/J0EYjoey/LLOT/blob/main/LLOT_util.py)


## Input:
LLOT requires input from two sources: a spatial transcriptomics dataset and  a scRNA-seq dataset.

## Example:
The jupyter notebook file [LLOT_example_Drosophila.ipynb](https://github.com/J0EYjoey/LLOT/blob/main/LLOT_example_Drosophila.ipynb) provides an example of how to use LLOT to obtain coupling matrix, and then predict spatial expressions of genes and infer cells' possible locations. 
