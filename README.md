# An Unrolled Half-Quadratic Approach for Sparse Signal Recovery in Spectroscopy


  
* **Authors**: Mouna Gharbi, Emilie Chouzenoux, Jean-Christophe Pesquet
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


    
### How to run

Make sure that PyTorch is available in your Python environment. The code has been tested with Python 3.8.5 and PyTorch 1.8.0.

In order to create data, please refer to the `.csv` file and use `create_data.py`.
To create a dataset named `Dataset1` with specified train, test and validation sizes run:
```bash
python ./Datasets/Create_Data.py --Dataset Dataset1 ..sth that works
```
In case the exact datasets used in the paper are 
required, contact me via email.

In order to run an experiment with the generated datasets, execute the 
respective Python scripts.
For example, unrolled half-quadratic goes as
```bash
python ./Unrolled_methods/U_HQ/runfile.py --Dataset Dataset1 ...
```
To see the exact command-line parameters available 
for each method check the corresponding files:

|Algorithm| Script to run|
|--|--|
|Unrolled Half-Quadratic|`Unrolled_methods/U_HQ/runfile.py`|...|
|Unrolled ISTA|`Unrolled_methods/U_ISTA/runfile.py`|


