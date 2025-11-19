# Hybrid Modeling Framework for Li-ion Battery State Prediction : A Physiscs-Informed Approach  
# By Ajin Thomas, 
This README.md file gives an idea about how to reproduce the codes on a new system.

## Acknowledgments
This repository builds upon and extends code from the original [PINN4SOH](https://github.com/wang-fujin/PINN4SOH) repository by wang-fujin. 
We have reorganized the project structure, consolidated scripts, and made modifications for our implementation.
We acknowledge and appreciate the original author's work and contributions. 

# 1. System requirements
python version: 3.9.21
| Package / Library | Version (tested) |
| :---------------: | :--------------: |
|       Python      |      3.9.21  |
|       torch       |       2.2.2      |
|       numpy       |      1.26.4      |
|       pandas      |       2.2.3     |
|    scikit-learn   |       1.6.1      |
|     matplotlib    |       3.9.2      |
|    scienceplots   |      latest      |
|        tqdm       |      latest      |


# 2. Installation
If you are not willing to use Python and Pytorch framework, the easiest way is to install Anaconda first and use Anaconda to quickly configure the environment.
## 2.1 Creating environment
conda create -n hypinn python=3.9.21
## 2.2 Activating environment
conda activate hypinn
## 2.3 Installing requirement 
pip install torch numpy pandas scikit-learn matplotlib scienceplots tqdm
Alternatively you can install by the followwing 
pip install -r requirements.txt
# Repository structure 
HYPINN/
│
├── data/
│   ├── Dataset_A/         # used for SOH prediction  
│   └── Dataset_B/         # for transfer learning 
│
├── Data_loading/
│   └── dataloader.py       # Dataset loaders for A & B
│
├── Model/
│   ├── Model.py            # Hybrid PINN architecture 
│   └── Comparing_Models.py # MLP & CNN architecture
│
├── main_scripts/
│   ├── PINN_main.py        # training script for Hybrid PINN
│   └── MLP_CNN_main.py     # training script for MLP and CNN
│
├── comparison/
│   ├── Dataset_A_metrics.py        # creates an .xlsx file with HYPINN evaluation metrics
│   ├── small_sample_PINN.py        # HYPINN under small sample regime
│   ├── small_sample_CNNandMLP.py   # CNN/MLP under small sample regime
│   ├── Model_comparison_table.py   # generates comparison tables
│
├── data analysis/
│   ├── correlation_analysis.py     # feature correlation heatmap generation
│   └── sample count.py             # counts total samples per datasets 
│
├── plotting_scripts/
│   ├── CapacityVsCycle.py          # plots battery capacity degradation
│   ├── plot_violin_rmse_clean.py   # RMSE violin plot for models
│   └── True_VS_Predicted.py        # scatter plots for true vs predicted SOH
│
├── transfer_L/
│   ├── transfer_utils.py           # transfer learning utilities 
│   └── transferL.py                # runs Dataset_A → Dataset_B adaptation
│
├── utils/
│   ├── util.py                     # metric computation, schedulers
│   ├── check_version.py            # environment check script
│   └── count_parameters.py         # prints parameter counts
│
├── results for analysis/           # all results comes under this for analysis
│
└── README.md 

# 3. A demo run 
A detailed demo for the code run. 
1. Firstly run the `PINN_main.py` file to train Hybrid PINN model. The code will generate results folder named `results for analysis` and save the results in it.
2. Run the `MLP_CNN_main.py` file. This file will also created outputs under `results for analysis` which are the results of the corresponding model (CNN or MLP).
3. If you run the `Dataset_A_metrics.py` file under the file `comparison`, It will process the results in step one and generate the `PINN-Dataset_A-results.xlsx` file under `results for analysis`. 
4. The file named `comparison` is with small sample learning codes. This is structuresd like the main files and if you run that, the code will produce the small sample results under `results for analysis` for each models.
5. `transfer_L` file contains the codes for transfer learning on the second dataset Dataset_B. 
6. All the plotting codes are filed under `plotting_scripts`.

Note: We know that the training process of neural network is random, and the volatility of regression models is normally greater than that of classification models. 
Therefore, the results obtained from the above process are not expected to be exactly identical to those mentioned in my thesis. 



