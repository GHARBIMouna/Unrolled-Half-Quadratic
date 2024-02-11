# An Unrolled Half-Quadratic Approach for Sparse Signal Recovery in Spectroscopy


  
  
* **Publication**: This is the source repository of the following work
  > Mouna Gharbi, Emilie Chouzenoux, Jean-Christophe Pesquet. An Unrolled Half-Quadratic Approach for Sparse Signal Recovery in Spectroscopy. Inria Saclay. 2023. ⟨hal-04229774⟩ 

* **Instituition**:  Centre de Vision Numérique, CentraleSupélec, Inria, Université Paris-Saclay
* **Email**: mouna.gharbi@centralesupelec.fr
  


    
### How to run

Make sure that PyTorch is available in your Python environment. The code has been tested with Python 3.8.5 and PyTorch 1.8.0.
All of the following commands are assumed to be 
run in the root directory of the repository.

In order to create data for training, please refer to the `.csv` file and use `create_data.py`.
To create a dataset named `Dataset1` with specified train, validation and test sizes run:
```bash
python ./Datasets/Create_Data.py --Dataset Dataset1 --set_size_training 900 --set_size_validation 100 --set_size_test 100
```
In case the exact datasets used in the paper are 
required, please contact me via email.

In order to run an experiment with the generated datasets, execute the 
respective Python scripts.
For example, to train unrolled half-quadratic (U-HQ)
```bash
python ./Unrolled_methods/U_HQ/runfile.py --Dataset Dataset1 --function train --architectre_lambda lamda_Acrh2_overparam
```
To see the exact command-line parameters available 
for each method check the corresponding files:

|Method| Script to run|
|--|--|
|Iterative Half-Quadratic|`Iterative_methods/HQ.py`|
|Unrolled Half-Quadratic|`Unrolled_methods/U_HQ/runfile.py`|
|Unrolled ISTA|`Unrolled_methods/U_ISTA/runfile.py`|
|Unrolled PD|`Unrolled_methods/U_PD/runfile.py`|
|FCNet|`Deep_learning_methods/FCNet_AE/FCNet_AE.py`|
|ResUNet|`Deep_learning_methods/ResUNet/ResUNet.py`|




### Detailed Files organization
The repository is organized as follows:

* `Datasets`

   *  `MSdata.csv`: contains original **Mass Spectrometry** data.
   *  `Create_Data.py`: creates datasets with different blur kernels and noise levels. A Dataset folder contains  respectively training, validation and test subfolders. Each of these folders contains a `Groundtruth`, `Degraded`, `Degraded2000` and `H` folders. To reproduce  `Dataset1` from the paper, create respectively $900$, $100$ and $100$ `training`, `validation` and `test` samples. `Groundtruth` signals $\overline{x}$ (of size $n=2000$) have a monoisotopic mass-to-charge ratio $m/z$ between $0$ and $200$ ppm. `Degraded` signals $y$ (of size m=2049) are created through a convolution (mode=Full) with a varying Ricker kernel `H` with spreading factor unifromly between $0.25$ and $1$. Finally, a standrad Gaussian noise, with a noise variance uniformly sampled between $0$ amd $0.5$ is added. `Degraded` is used for unrolled, FCNet and AE trainings. `Degraded2000` (convolution mode=same) is used for ResUNet training.
  Dataset2 contains respectively $900$, $100$ and $100$ `training`, `validation` and `test` samples. `Groundtruth` signals $\overline{x}$ (of size $n=2000$) have a monoisotopic mass-to-charge ratio $m/z$ between $0$ and $200$ ppm. `Degraded` signals $y$ (of size m=2049) are created through a convolution (mode=Full) with a varying Ricker kernel `H` with spreading factor unifromly between $0.25$ and $1$. Finally, a standrad Gaussian noise, with a noise variance uniformly sampled between $0.5$ amd $1$ is added. `Degraded` is used for unrolled, FCNet and AE trainings. `Degraded2000` (convolution mode=same) is used for ResUNet training.Dataset3 contains respectively $900$, $100$ and $100$ `training`, `validation` and `test` samples. `Groundtruth` signals $\overline{x}$ (of size $n=2000$) have a monoisotopic mass-to-charge ratio $m/z$ between $0$ and $200$ ppm. `Degraded` signals $y$ (of size m=2049) are created through a convolution (mode=Full) with a varying Fraser Suzuki kernel `H` with spreading factor unifromly between $0.25$ and $1$ and an asymmetry factor between $0.2$ and $0.6$. Finally, a standrad Gaussian noise, with a noise variance uniformly sampled between $0$ amd $0.5$ is added. `Degraded` is used for unrolled, FCNet and AE trainings. `Degraded2000` (convolution mode=same) is used for ResUNet training.

* `Deep_Learning_Methods`
    * `FCNet_AE`: trains and tests  fully connected and autoencoder-like architectures.
    * `ResUNet`: trains and tests a residual UNet architecture as in the code in [this repository ](https://github.com/conor-horgan/DeepeR.git).
* Iterative methods:
    * `HQ.py: performs grid search and tests Half-Quadratic variants HQ-SC and HQ-ES.

* Unrolled methods:
    * `U_HQ`: trains and tests all variants of the unrolled Half-Quadratic paradigm: U-HQ-DE, U-HQ-FixS, U-HQ-FixN and U-HQ-FixN-Over-P.
    * `U_PD`: trains and tests unrolled primal-dual algorithm.
    * `U_ISTA`: trains and tests unrolled iterative soft thresholding algorithm.

* Chromatography_Toolbox: This is a tool that allows us to build simulated chromatographic datasets. This tool was not used in the experiments of the aforementioned paper but it can be useful to provide simulated datasets for the user. For more information please check the 

### Acknowle>dgements
The comparisons with the ResUNet method are based on the code of Conor C. Horgan (see [github repo](https://github.com/conor-horgan/DeepeR.git)) and all experiments are run on mass spectrometry data from [MassBank](https://massbank.eu/MassBank/Search).

This work has been supported by the ITN-ETN project TraDE-OPT funded
by the European Union's Horizon 2020 research and innovation programme
under the Marie Sklodowska-Curie grant agreement No 861137 and by
the European Research Council Starting Grant MAJORIS ERC-2019-STG850925.


