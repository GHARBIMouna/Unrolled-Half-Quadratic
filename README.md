# An Unrolled Half-Quadratic Approach for Sparse Signal Recovery in Spectroscopy


* **License**: 
* **Author**: Mouna Gharbi
* **Instituition**:  Centre de Vision Numérique, CentraleSupélec, Inria, Université Paris-Saclay
* **Email**: mouna.gharbi@centralesupelec.fr
* **Related Publication**: This is the source repository of the following work
  > Mouna Gharbi, Emilie Chouzenoux, Jean-Christophe Pesquet. An Unrolled Half-Quadratic Approach for Sparse Signal Recovery in Spectroscopy. Inria Saclay. 2023. ⟨hal-04229774⟩ 




### Dependencies
Python 3.8.5  
Pytorch 1.8.0

### Files organization
* `Datasets`: This folder contains `Dataset1` (README), `Dataset2` (README) and `Dataset3` (README). Each dataset Contains respectively 900, 100, and 100 training, validation and test pairs in the corresponding folders `training`, `validation` and `test` set. Each of these folders contains a `Groundtruth`, `Degraded`, `Degraded2000` and `H` folders. 
    * `Dataset1`
    * `Dataset2`
    * `Dataset3`
   *  `MSdata.csv`: contains original data.
   *  `Create_Data.py`: creates datasets with different blur kernels and noise levels.

* `Deep_Learning_Methods`
    * `FCNet_AE`: trains and tests  fully connected and autoencoder-like architectures.
    * `ResUNet`: trains and tests a residual UNet architecture as in the code [[github](https://github.com/conor-horgan/DeepeR.git)].
* Iterative methods:
    * `Half_Quadratic.py`: performs grodsearch and tests Half-Quadratic variants HQ-SC and HQ-ES.

* Unrolled methods:
    * `U_HQ`: trains and tests  all variants of the unrolled Half-Quadratic paradigm: U-HQ-DE, U-HQ-FixS, U-HQ-FixNand U-HQ-FixN-OverP.
    * `U_PD`: trains and tests unrolled primal dual algorithm.
    * `U_ISTA`: trains and tests unrolled iterative soft threshholding algorithm.


    
### Quick demo

`demo.ipynb`: Tests pre-trained U-HQ and illustrates how to perform training.

