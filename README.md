# deep_metab

Applying Deep Learning Methods to LC-MS Metabolomics Data to Improve Metabolite Identification

# TO RUN ON HPC:
### DO NOT MOVE .sif FILES !


Do not change .sif name and take caution editing the following paths, as they are binded:
```
workspace/Graphormer-RT/checkpoints/
graphormer_checkpoints/
```

```
$ apptainer build graphormercontainer.sif docker://dnhem/proj_deepmetab:v0.1 # NOTE VERSION
$ mkdir -p predictions

### IF TRAINING ###
#BROKEN -> $ sbatch runmodel_apptainer.sh # KC WILL FIX

### IF USING PRESET ###
### USE THIS ONE ###
$ mkdir -p graphormer_checkpoints && wget https://zenodo.org/records/15021743/files/oct30_RP_unc.pt?download=
1 -O graphormer_checkpoints/oct30_RP_unc.pt 


```



