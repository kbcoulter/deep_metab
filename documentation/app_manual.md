# Step-by-Step User Manual - No Root Access

> **⚠️ This project is under active development. Expect rapid changes.**

> **Note: We expect to pass filepaths as arguments rather than setting paths within scripts. When this change is complete, this manual will be updated accordingly.**

## Requirements
**Please ensure that both SLURM and Apptainer are available on your device.** You can check this by running the following and ensuring that ```$: command not found``` is *not* returned:
```bash
apptainer --version
scancel
```

## Table of Contents
 - [Requirements](#requirements)
 - [Installation](#installation)
 - [Setup](#setup)
 - [Data Preparation](#data-preparation)
    - [LC-MS Metadata](#lc-ms-setup-information-metadata)
    - [LC-MS Prediction Data](#lc-ms-prediction-datasets)
    - [Data Location](#data-location)
- [Finetuning Model Weights (Optional)](#finetuning-model-weights-optional)
    - [Splitting Data](#first-please-split-your-data)
 - [Data Assumptions](#data-assumptions)
 - [Making RT Predictions](#making-rt-predictions)
 - [Evaluating Predictions](#evaluating-predictions)
 - [Applying Scoring Frameworks](#applying-scoring-frameworks)
 - [Extra Scripts (Currently No Support)](#extra-scripts)


---


## Installation
First, clone the repo from source and navigate into it:

```bash
git clone https://github.com/kbcoulter/deep_metab.git
cd deep_metab
```

## Setup
 From ```deep_metab```:
 1. Setup the git submodule, 
 2. Build your container: ```graphormercontainer.sif```, 
 3. Setup your necessary directories 
 
 All 3 steps can be completed by running one of the following to setup for RP, HILIC, or Both, respectively:

```bash
./setup_HPC/RP.sh
./setup_HPC/HILIC.sh
./setup_HPC/HILIC_RP.sh
```

Note: Options for model weights and container weights can be edited within these scripts.

## Data Preparation

### LC-MS Setup Information Metadata:

```bash
```

### LC-MS Prediction Datasets:

```bash
```

### Data Location 
To simplify your experience, we highly reccomend placing your data within a new directory here: ```my_data\<dir_created>```

## Finetuning HILIC Model Weights (Optional)
This step is entirely optional, but reccomended if you have **HILIC** data with validated retention times to train with.


> **WARNING: This process will write over ```best_checkpoint.pt``` and ```last_checkpoint.pt``` should they exist in ```graphormer_checkpoints_HILIC```.** 

### First, please split your data. 
This can be done using by reading model options and subsequently running the following:

```bash
./data_prep/finetune_HILIC_split.py -h
./data_prep/finetune_HILIC_split.py -d <data> -r <seed> -s <split> -o <out>
```
Then, please move that data into ```my_data/HILIC_ft/```

### To Finetune:
```bash
sbatch ./setup_model/app_finetune_HILIC.sh
```
Note: You may need to change the data paths in the script before running. 

## Data Assumptions:
- It is absolutely essential that RP and HILIC data **do not** live in the same directory.
- Currently, we assume one metadata.pkl file in each .csv file directory. 
    - This file can be identical, thanks to the setupID code.

## Making RT Predictions

#### To make a RT predictions run one of the following scripts to make RP or HILIC predictions, respectively:
```bash
sbatch ./make_predictions/app_evaluate_RP.sh
sbatch ./make_predictions/app_evaluate_HILIC.sh
```
Note: You will likely need to change the data paths in the script before running. 

## Evaluating Predictions
```bash
```

## Applying Scoring Frameworks
```bash
```

## Extra Scripts
#### Please note: 
These scripts were used in the creation of this repository and to complete associated analyses. They are not omptimized for user experience and may include multiple hard paths and/or issues. They are included only in the hopes that they save you time and/or effort in your application of this tool.

Currently, we do not offer support for these scripts. This may change in the near future... 

| Script                         | Purpose                                                                 |
|--------------------------------|-------------------------------------------------------------------------|
| extra_scripts/training_loss.py    | Generates a `.png` plot to visualize model training and validation loss by epoch |
| extra_scripts/join_HILIC_ichikey.Rmd | Joins 2 `.csv` pairs on ichikey, scraps missing, and takes the middle of expected RT range for finetuning dataset. |
|extra_scripts/HILIC_knowns_comp.Rmd| Generates predicted vs expected RT overlaid predicted vs observed RT `.png` and a prediction absolute difference `.png`|


