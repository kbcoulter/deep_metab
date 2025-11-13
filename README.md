# deep_metab

Applying Deep Learning Methods to LC-MS Metabolomics Data to Improve Metabolite Identification

# TO RUN ON HPC:
### DO NOT MOVE .sif FILES !


Do not change .sif name and take caution editing the following paths, as they are binded:
```
workspace/Graphormer-RT/checkpoints_RP/
graphormer_checkpoints_RP/
workspace/Graphormer-RT/checkpoints_HILIC/
graphormer_checkpoints_HILIC/
```

## FOR ALL MODELS
#### Please build the [docker](https://hub.docker.com/r/dnhem/proj_deepmetab) image made available online.
```
$ apptainer build graphormercontainer.sif docker://dnhem/proj_deepmetab:v0.1 # NOTE VERSION
```

## Reverse Phase (RP) ###
#### TRAINING CURRENTLY BROKEN, RUN PRETRAINED

### 1A. IF TRAINING NEW MODEL ###
```
sbatch runmodel_apptainer.sh&& \
mkdir -p predictions_RP 
```

### 1B. IF USING PRETRAINED ###
```
mkdir -p graphormer_checkpoints_RP &&\
wget -O graphormer_checkpoints_RP/oct30_RP_unc.pt https://zenodo.org/records/15021743/files/oct30_RP_unc.pt?download=1 &&\
mkdir -p predictions_RP
```

### 2. EVALUATE / MAKE PREDICTIONS ###
After editing options, run:
```
$ sbatch app_evaluate_RP.sh 
``` 

## HILIC ##
```
### CREATE PREDICTION FOLDER ###
$ mkdir -p predictions_HILIC

### MAKE EVALUATIONS ###
$ sbatch app_evaluate_HILIC.sh # EDIT OPTIONS
```

## OUTPUT FOR PREDICTIONS
Output from successful evaluation/ generation of predictions should contain some messages from the creators of Graphormer-RT along with a new print statement: 
```
SAVED LC-MS-TYPE PREDICTIONS
```