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

## SETUP ##
#### Run the appropriate setup script from within the setup_HPC directory, depending on your LC-MS configuration:
```
./RP.sh
./HILIC.sh
./HILIC_RP.sh
```

#### If you encounter errors, check the available image versions on [Docker Hub](https://hub.docker.com/r/dnhem/proj_deepmetab).

Note: Evaluation with personally trained models is currently buggy, please use pretrained models. We are working to fix this issue. 


## EVALUATION / RT PREDICTION ##
### RP ###
```
$ sbatch app_evaluate_RP.sh # EDIT OPTIONS
``` 

### HILIC ###
```
$ sbatch app_evaluate_HILIC.sh # EDIT OPTIONS
```

## OUTPUT FOR PREDICTIONS
Output from successful evaluation/ generation of predictions should contain some messages from the creators of Graphormer-RT along with a new print statement: 
```
SAVED LC-MS-TYPE PREDICTIONS
```