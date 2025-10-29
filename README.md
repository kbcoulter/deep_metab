# deep_metab

Applying Deep Learning Methods to LC-MS Metabolomics Data to Improve Metabolite Identification

# TO BUILD CONTAINER ON HPC:
### DO NOT MOVE .sif FILES !

Edit the slurm options in runmodel_apptainer.sh,
but do not change .sif name:

```
$ apptainer build graphformercontainer.sif docker://dnhem/proj_deepmetab:v0.0

$ sbatch runmodel_apptainer.sh 

```



