# Step-by-Step User Manual - No Root Access

> **⚠️ This project is under active development. Expect rapid changes.**

> **Note: We expect to pass filepaths as arguments rather than setting paths within scripts. When this change is complete, this manual will be updated accordingly.**

## Requirements
**Please ensure that both SLURM and Apptainer are available on your device.** You can check this by running the following and ensuring that ``command not found`` is *not* returned:
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

The metadata pickle encodes your chromatography setup (column, gradient, etc.) and must sit in the **same directory** as your prediction CSV. Use one of the following approaches.

**Preliminary — RepoRT-style processed data**
First, you must have 3 per-sample directories with `{id}_gradient.tsv`, `{id}_metadata.tsv`, and `{id}_info.tsv` configured:

```bash
./data_prep/process_experimental_conditions/gen_pickle_dict.py \
  --data_directory <path_to_processed_data> \
  -o <output_dir> \
  -of metadata.pickle
```

Use `--tanaka` and `--hsmb` to override the default Tanaka/HSMB database paths if needed. If your metadata TSV format does not match the expected template, run `reformat_meta.py` first:

```bash
./data_prep/process_experimental_conditions/reformat_meta.py -i <metadata.tsv> -d <output_dir> -id <sample_id>
```

### LC-MS Prediction Datasets:

Prepare a single CSV in Graphormer-RT format (`setup_id,smiles,averaged_retention_time`, no header) and place it in the **same directory** as your metadata pickle.

**1. Aggregate your LC–MS CSVs** into one file (File Name, Mass Feature ID, Retention Time, SMILES):

```bash
./data_prep/process_lcms_data/initiate_dataset.py \
  --dataset_folder <path_to_csvs> \
  --output_csv <aggregated.csv> \
  --mass_feature_id_col "Mass Feature ID" \
  --retention_time_col "Retention Time (min)" \
  --smiles_col "smiles"
```

**2. (Optional) Remove stereochemistry** from the SMILES column. If you do this, use the output as input to step 3 and set `--smiles_col` to the 0-based index of the destereoed SMILES column (e.g. `4` if `smiles_destereo` is the 5th column).

```bash
./data_prep/process_lcms_data/rm_stereochemistry.py \
  -i <aggregated.csv> -o <aggregated_destereo.csv> \
  --smiles-col-index 3 --has-header
```

**3. Build Graphormer-RT input** (average RT per unique SMILES, add setup ID):

```bash
./data_prep/process_lcms_data/initiate_input_for_graphormer_eval.py \
  -i <aggregated.csv> -o <graphormer_input.csv> \
  -s <setup_id>
```

Use the output from step 2 as `-i` if you ran the optional destereo step. Use `--file_name_col`, `--retention_time_col`, and `--smiles_col` if your column positions differ (defaults 0, 2, 3). Place `graphormer_input.csv` and your metadata pickle together in e.g. `my_data/<dir_created>/`, then set `HOST_DATA_DIR` in the eval script to that directory.

### Data Location 
To simplify your experience, we highly recommend placing your data within a new directory here: ```my_data/<dir_created>```

## Finetuning HILIC Model Weights (Optional)
This step is entirely optional, but recommended if you have **HILIC** data with validated retention times to train with.


> **WARNING: This process will write over ```best_checkpoint.pt``` and ```last_checkpoint.pt``` should they exist in ```graphormer_checkpoints_HILIC```.** 

### First, please split your data. 
This can be done by reading model options and subsequently running the following:

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

#### To make RT predictions, run one of the following scripts to make RP or HILIC predictions, respectively:
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
These scripts were used in the creation of this repository and to complete associated analyses. They are not optimized for user experience and may include multiple hard paths and/or issues. They are included only in the hopes that they save you time and/or effort in your application of this tool.

Currently, we do not offer support for these scripts. This may change in the near future…

| Script | Purpose |
|--------|---------|
| `extra_scripts/training_loss.py` | Parses a SLURM/training log (e.g. `tlearn_HILIC_*.out.log`) for train/valid loss per epoch and generates a `.png` plot of training vs validation loss. **Hardcoded log path**—edit the script before use. |
| `extra_scripts/join_HILIC_ichikey.Rmd` | Joins two CSVs (`LD.csv`, `RD.csv`) on `inchikey`; filters missing values; aggregates min/max RT by compound; computes midpoint RT. Writes `HILIC_KNOWNS.csv` for HILIC finetuning. |
| `extra_scripts/HILIC_knowns_comp.Rmd` | Reads `HILIC_KNOWNS.csv` and prediction CSVs. Pairs predictions with knowns, then produces a predicted vs observed RT scatter (knowns + ambiguous), a diagonal reference line, and saves `Pred_vs_Obs.svg`. Also computes MAE and runs a Wilcoxon test. **Hardcoded input filenames.** |
| `extra_scripts/destereo_stats.py` | Compares original SMILES (`output.csv`) vs destereoed SMILES (`data_without_stereo.csv`). Reports unique counts, reduction from stereochemistry collapse vs canonicalization, collapse-size distribution, and example collapses. **Hardcoded paths.** |
| `extra_scripts/download_data.py` | Reads a CSV of `(data_id, url)` rows and downloads each URL into an output directory as `{data_id}_{counter:04d}`. **Hardcoded input CSV and output directory** (e.g. HILIC pos/neg). |
| `extra_scripts/pickle_to_csv.py` | Converts a metadata pickle (Python dict) to CSV. `-i` / `--input`: pickle path; `-o` / `--output`: CSV path (default `cleaned_data.csv`). |
| `extra_scripts/quick_analyze_predictions.py` | Reads a predictions CSV (columns: SMILES, …, True RT, Predicted RT). Computes absolute and percent difference; writes a report with counts and proportions in 10% bins (≤10%, 10–20%, …, 40–50%) and lists SMILES per bin. `-i` / `--input`: predictions file; `-o` / `--output`: report path (default `difference.txt`). |
| `extra_scripts/unpack_pickle.py` | Loads a pickle file and prints its contents to stdout. Usage: `unpack_pickle.py <path_to_pickle>`. |


