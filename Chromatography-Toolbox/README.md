This tool was not used in the published paper but can be useful to create simulated chromatographic datasets. 
This code assumes that the datasets have a **constant** blur kernel. Please adapt according to experimental needs.
### How to run
All of the following commands are assumed to be 
run in the root directory of the repository.

In order to create data for training named `Dataset1`, provide the training, validation and test set sizes `N_tr`, `N_val` and `N_test` respectively. Provide also the Fraser Suzuki asymmetry coefficient `a`, the number of spikes determined by `percentage`, the distance between two spikes `peak_inter_dist` and the noise level `sG`.  Please run 

```bash
python ./Chromatography_Toolbox/Create_Dataset.py --Dataset Dataset1 --N_tr 900 --N_val 100 --N_test 100 --a 0.2 --percentage 0.003 --sG 0.02 --peak_inter_dist 3
```

In order to use **HALmetrics**, the chemically-driven metrics that rely on the determination of height, location and area of predicted peak, we assume the user has access to a folder `Rec_Signals` containing predicted signals. please refer to `Evaluate_Halmetrics.py` to give the correct paths. Run

```bash
python ./Chromatography_Toolbox/Evaluate_Halmetrics.py 
```