# Step-by-Step User Manual - No Root Access

> **⚠️ This project is under active development. Expect rapid changes.**

# Table of Contents

🚀 **Getting Started**
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Setup](#setup)

🧪 **Data Preparation**
  - [LC-MS Metadata](#lc-ms-setup-information-metadata)
  - [LC-MS Prediction Data](#lc-ms-prediction-datasets)
  - [Data Location](#data-location)
  - [Data Assumptions](#data-assumptions)

🤖 **Prediction and Scoring**
  - [Making RT Predictions](#making-rt-predictions)
  - [Evaluating Predictions (Optional)](#evaluating-predictions)
  - [Applying Scoring Frameworks](#applying-scoring-frameworks)

🧰 **Advanced & Extras**
  - [Fine-tuning Model Weights (Optional)](#finetuning-hilic-model-weights-optional)
    - [Splitting Data](#first-please-split-your-data)
- [Extra Scripts (No Current Support)](#-extras)



###
# 🚀 **Getting Started**
## Requirements
**Please ensure that both SLURM and Apptainer are available on your device.** You can check this by running the following and ensuring that ``command not found`` is *not* returned:
```bash
apptainer --version
scancel
```

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
 4. Download your model weights (into checkpoint directories)
 
 All 4 steps can be completed by running one of the following to setup for RP, HILIC, or Both, respectively.

```bash
./setup_HPC/RP.sh \
  --version-custom <version> \           # Default: v0.3
  --image-custom <image> \               # Default: dnhem/proj_deepmetab
  --rweights-custom <rp_weights_url> \   # Default: RP weights

./setup_HPC/HILIC.sh \
  --version-custom <version> \          # Default: v0.3
  --image-custom <image> \              # Default: dnhem/proj_deepmetab
  --hweights dm|og | --hweights-custom <hilic_weights_url>

./setup_HPC/HILIC_RP.sh \
  --version-custom <version> \          # Default: v0.3
  --image-custom <image> \               # Default: dnhem/proj_deepmetab
  --rweights-custom <rp_weights_url> \   # Default: RP weights
  --hweights dm|og | --hweights-custom <hilic_weights_url>
```
 #### To confirm setup, please ensure that the configuration shown in the terminal matches expected.
> **Note:** We recommend leaving options with a default setting unchanged. For `--hweights` (HILIC weights) please specify `dm` (deep_metab), `og` (original Graphormer-RT), or provide your own with `--hweights-custom <custom_url>.` 
###
# 🧪 **Data Preparation**
## Data Preparation

### LC-MS Setup Information Metadata:

The metadata pickle encodes your chromatography setup (column, gradient, etc.) and must sit in the **same directory** as your prediction csv. Use one of the following approaches:

**Preliminary — [RepoRT](https://github.com/michaelwitting/RepoRT/)-style processed data**. First, you must have 3 directories per-sample with `{id}_gradient.tsv`, `{id}_metadata.tsv`, and `{id}_info.tsv` configured:

```bash
./data_prep/process_experimental_conditions/gen_pickle_dict.py \
  --data_directory <path_to_processed_data> \
  -o <output_dir> \
  -of metadata.pickle
```

Use `--tanaka` and `--hsmb` to override the default Tanaka/HSMB database paths if needed. If your metadata TSV format does not match the expected template, run `reformat_meta.py` first:

```bash
./data_prep/process_experimental_conditions/reformat_meta.py \
-i <metadata.tsv> \
-d <output_dir> \
-id <sample_id>
```

### LC-MS Prediction Datasets:

Prepare a single CSV in Graphormer-RT format (`setup_id,smiles,averaged_retention_time`) with no header and place it in the **same directory** as your metadata pickle.

**1. Aggregate your LC–MS CSVs** into one file (File Name, Mass Feature ID, Retention Time, SMILES):

```bash
./data_prep/process_lcms_data/initiate_dataset.py \
  --dataset_folder <path_to_csvs> \
  --output_csv <aggregated.csv> \
  --mass_feature_id_col "Mass Feature ID" \
  --retention_time_col "Retention Time (min)" \
  --smiles_col "smiles"
```

**2. Build Graphormer-RT input** (average RT per unique SMILES, add setup ID):

```bash
./data_prep/process_lcms_data/initiate_input_for_graphormer_eval.py \
  -i <aggregated.csv> -o <graphormer_input.csv> \
  -s <setup_id>
```

Use `--file_name_col`, `--retention_time_col`, and `--smiles_col` if your column positions differ (defaults 0, 2, 3). Place `graphormer_input.csv` and your metadata pickle together in e.g. `my_data/<dir_created>/`, then set `HOST_DATA_DIR` in the eval script to that directory.

### Data Location 
To simplify your experience, we **highly recommend** placing your data within a new directory here: ```my_data/<dir_created>```


## Finetuning HILIC Model Weights (Optional)
### 🧰 **Advanced**
This step is entirely optional, but recommended if you have **HILIC** data with validated retention times to train with.


> **WARNING: This process will write over ```best_checkpoint.pt``` and ```last_checkpoint.pt``` should they exist in ```graphormer_checkpoints_HILIC```.** 

### First, please split your data. 
This can be done by reading model options and subsequently running the following:

```zsh
./data_prep/finetune_HILIC_split.py \
  -d <data> \
  -r <seed> \
  -s <split> \
  -o <out>
```
Then, please move that data into ```my_data/HILIC_ft/```

### To Finetune:
```bash
sbatch ./setup_model/app_finetune_HILIC.sh \
  --data-file <path_to_csv> \ 
  --metadata-file <path_to_pickle> \
  --user-data-dir <path>              # Default: my_data/HILIC_ft
  --batch-size <int>                  # Default: 48
  --max-epoch <int>                   # Default: 100
  --checkpoint-dir <path>             # Default: /graphormer_checkpoints_HILIC/
  --pretrained-model <path>           # Default: /graphormer_checkpoints_HILIC/HILIC_WEIGHTS.pt
```
> **Note:** These paths are bind mounted. The true path being accessed is
`/workspace/Graphormer-RT/checkpoints_HILIC/HILIC_WEIGHTS.pt`

Additional Options Include: seed, attention-dropout, act-dropout, dropout, adam-betas, adam-eps, clip-norm, weight-decay, learning rate (lr), freeze-level, and more. To edit these, please manually change them within the script `app_finetune_HILIC.sh`. It is possible that support for in line options will be added in a future release.




## Data Assumptions:
> **NOTE 1:** Please ensure that both RP and HILIC raw datasets are organized in their own directories.

>**NOTE 2:** metadata.pkl file exists in each .csv file directory. This file can be identical, thanks to the setupID code.

###
# 🤖 **Prediction and Scoring**

## Making RT Predictions

#### To make RT predictions, run one of the following scripts to make RP or HILIC predictions, respectively:
```bash
sbatch ./make_predictions/app_evaluate_RP.sh \
  --host-data-dir <path> \   # Default: /my_data/sample_data_0001/
  --checkpoint-dir <path> \  # Default: /graphormer_checkpoints_RP
  --save-path <path>         # Default: /predictions_RP

sbatch ./make_predictions/app_evaluate_HILIC.sh \
  --host-data-dir <path> \   # Default: /my_data/HILIC_Posttraining/
  --checkpoint-dir <path> \  # Default: /graphormer_checkpoints_HILIC
  --save-path <path>         # Default: /predictions_HILIC
```
> **Note:** The checkpoint paths are bind mounted. The true paths being accessed are
`/workspace/Graphormer-RT/checkpoints_HILIC`

>**Note:** The script will search the checkpoint directory for model weights (for ease of use), please remove or hide unwanted model weights. 

## Evaluating Predictions
After running your predictions, ensure that they exist in the correct directory `predictions_HILIC` or `predictions_RP`. For a quick check, you can run the following to get a quick overview report of the predictions made by Graphormer-RT.
```bash
./data_prep/evaluate_rt_preds/analyze_predictions.py \ 
  -i <predictions.csv> \ 
  -o <output_name_report.txt> 
```

## Applying Scoring Frameworks

> **NOTE 1:** If you have validated observations of HILIC data, please ensure you've ran the model predictions following the steps outlined in [Making RT Predictions](#making-rt-predictions). 

> **NOTE 2:** Ensure you ran `./rm_stereochemistry.py` to generate the reference stereo files.
```bash
./data_prep/process_lcms_data/rm_stereochemistry.py \
  -i <aggregated.csv> \
  -o <aggregated_destereo.csv> \
  --smiles-col-index 3 \
  --has-header
```

A quick reminder that your directory tree should look something like this before running this script:
```text

deep_metab/
├── my_data/
│   ├── all_RP_raw_data/
│   │   ├── rp_1.csv
│   │   ├── rp_2.csv
│   │   └── ...
│   ├── all_HILIC_raw_data/
│   │   ├── hilic_1.csv
│   │   ├── hilic_2.csv
│   │   └── ...
│   ├── HILIC_knowns/
│   │   ├── hilic_knowns.csv
│   │   └── hilic_knowns_pred.csv
│   └── HILIC_ft/ # If You Have
│       ├── DM_Finetune.py
│       ├── featurizing_helpers.py
│       ├── HILIC_metadata.pickle
│       ├── hilic_finetuning_test.csv
│       ├── hilic_finetuning_train.csv
│       └── ...
├── predictions_HILIC/ 
│   └── hilic_preds.csv
├── predictions_RP/ 
│   └── RP_preds.csv
├── Destereo_Files/ # If You Have
│   ├── hilic_destereo.csv/
│   └── rp_destereo.csv
└── ...
```

This script appends predicted retention times (RTs) to LC-MS annotation files and/or performs post-hoc analysis on existing files. It supports both HILIC and RP chromatography workflows, with optional RT plausibility filtering and stereochemistry flagging. The goal of this step is to uniquely identify annotations from ambiguous annotations.

The full downstream workflow goes like this:
1. **RT Flagging** (Only if validated observations (knowns) are provided)
2. **Stereochemistry Stripping** (Must provide a reference data where smiles with stereochemistry stripped.)
3. **Apply Scoring Framework**

### Required Arguments
| Flag            | Description                                                                    |
| --------------- | ------------------------------------------------------------------------------ |
| `-i`, `--input` | Input directory containing LC-MS annotation files (raw or previously appended) |
| `-t`, `--type`  | Chromatography type: `hilic` or `rp`                                           |
| `--stereo_file` | Destereochemistry reference file used for stereo handling                      |
| `-p`, `--preds`  | CSV containing SMILES and predicted RTs (required for append mode) |

### Modes of Operation
The script can be run in different modes depending on which flags are provided. The table below summarizes how each mode behaves.
| Mode                     | Flags Required            | What It Does                                                    | Writes Files?          |
| ------------------------ | ------------------------- | --------------------------------------------------------------- | ---------------------- |
| **Append only**          | `-a`, `-p`                | Appends predicted RTs (lookup by SMILES) to each input file     | ❌ No (unless `--save`) |
| **Analyze only**         | `-s`                      | Computes ambiguity metrics and RT statistics and prints results | ❌ No                   |
| **Append + Analyze**     | `-a`, `-p`, `-s`          | Appends predicted RTs **and** reports summary statistics        | ❌ No (unless `--save`) |
| **RT-filtered analysis** | `-s`, `--rt_filter`, `--knowns`, `--knowns_pred`       | Applies RT plausibility filtering before computing statistics   | ❌ No (unless `--save`) |
| **RT-filtered append**   | `-a`, `-p`, `--rt_filter`, `--knowns`, `--knowns_pred` | Appends RTs and filters implausible annotations                 | ❌ No (unless `--save`) |

### RT Flagging (Optional)
> **Note:** Only use this flag if KNOWNS are provided! Currently, only HILIC KNOWNS are provided, so our function will only support HILIC data at the moment.

| Flag            | Description                                                   |
| --------------- | ------------------------------------------------------------- |
| `--rt_filter`   | Enable RT plausibility filtering                              |
| `--k_std`       | Standard-deviation multiplier for RT window (default: `2.0`)  |
| `--min_calib_n` | Minimum number of calibration points required (default: `50`) |

### Stereochemistry Handling
| Flag            | Description                                                      |
| --------------- | ---------------------------------------------------------------- |
| `--flag_stereo` | Flag rows where stereochemistry is present (`is_stereo == True`) |

### Other Optional Arguments
| Flag             | Description                                                        |
| ---------------- | ------------------------------------------------------------------ |
| `--save`                  | Enables writing appended and/or filtered outputs to disk        |
| `--knowns`       | File containing known compounds for RT calibration (optional)      |
| `--knowns_preds` | Predicted RTs for known compounds (used for calibration)           |

### Full Workflow Usage (HILIC w/ Knowns)

```bash
./data_prep/eval_rt_preds/filter_and_scoring.py \
  -i <all_HILIC_raw_data>/ \
  -p predictions_HILIC/<hilic_preds.csv> \
  -t hilic \
  -a \
  --knowns ../HILIC_KNOWNS/<HILIC_KNOWNS_DATA.csv> \
  --knowns_preds ../HILIC_KNOWNS/<HILIC_KNOWNS_PREDS.csv> \
  --save \
  --stereo_file <Destereo_Files/hilic_destereo> \ 
  --flag_stereo \
  --rt-filter

```

### Full Workflow Usage (HILIC w/o Knowns)

```bash
./data_prep/eval_rt_preds/filter_and_scoring.py \
  -i <all_HILIC_raw_data>/ \
  -p predictions_HILIC/<hilic_preds.csv> \
  -t hilic \
  -a \
  --save \
  --stereo_file <Destereo_Files/hilic_destereo> \ 
  --flag_stereo

```

### Full Workflow Usage (RP w/o Knowns)

```bash
./data_prep/eval_rt_preds/filter_and_scoring.py \
  -i <all_RP_raw_data>/ \
  -p predictions_RP/<RP_preds.csv> \
  -t rp \
  -a \
  --save \
  --stereo_file <Destereo_Files/rp_destereo> \ 
  --flag_stereo

```
###
# 🧰 **Extras**
These scripts were used in the creation of this repository and to complete associated analyses specific to our goals. They are not optimized for user experience and may include multiple hard paths, issues, or assumptions that do not translate to your use cases. They are included only in the hopes that they save you time and/or effort in your application of this tool. These scripts can all be found in `extra_scripts/` Command-line interface scripts that may be of great use are marked with a star: ⭐️ 

> **Note**  We do not offer support for these scripts. 

| Script | Purpose |
|--------|---------|
| `destereo_stats.py` | Compares original SMILES (`output.csv`) vs destereoed SMILES (`data_without_stereo.csv`). Reports unique counts, reduction from stereochemistry collapse vs canonicalization, collapse-size distribution, and example collapses. **Hardcoded paths.** |
| `finetuning_residuals.R` | Generates a Residual Plot from a created HILIC Finetuning csv file. **Hardcoded predictions input file** |
| `HILIC_knowns_comp.R` | Reads `HILIC_KNOWNS.csv` and prediction CSVs. Pairs predictions with knowns, then produces a predicted vs observed RT scatter (knowns + ambiguous), a diagonal reference line, and saves `Pred_vs_Obs.svg`. Also computes MAE and runs a Wilcoxon test. **Hardcoded input filenames.** |
| `join_HILIC_ichikey.R` | Joins two CSVs (`LD.csv`, `RD.csv`) on `inchikey`; filters missing values; aggregates min/max RT by compound; computes midpoint RT. Writes `HILIC_KNOWNS.csv` for HILIC finetuning. **Hardcoded data paths (LD & RD)** |
| `⭐️ download_data.py` | Reads a CSV of `(data_id, url)` rows and downloads each URL into an output directory as `{data_id}_{counter:04d}`. **Hardcoded input CSV and output directory** (e.g. HILIC pos/neg). |
| `⭐️ pickle_to_csv.py` | Converts a metadata pickle (Python dict) to CSV. `-i` / `--input`: pickle path; `-o` / `--output`: CSV path (default `cleaned_data.csv`). |
| `⭐️ quick_analyze.py` | Reads a predictions CSV (columns: SMILES, …, True RT, Predicted RT). Computes absolute and percent difference; writes a report with counts and proportions in 10% bins (≤10%, 10–20%, …, 40–50%) and lists SMILES per bin. `-i` / `--input`: predictions file; `-o` / `--output`: report path (default `difference.txt`). |
| `training_loss.py` | Parses a SLURM/training log (e.g. `tlearn_HILIC_*.out.log`) for train/valid loss per epoch and generates a `.png` plot of training vs validation loss. **Hardcoded log path** |
| `⭐️ unpack_pickle.py` | Loads a pickle file and prints its contents to stdout. Usage: `unpack_pickle.py <path_to_pickle>`. |


