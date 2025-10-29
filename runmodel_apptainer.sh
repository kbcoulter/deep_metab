#!/bin/bash

#SBATCH --account=kcoulter
#SBATCH --partition=gpulong
#SBATCH --job-name=deep_metab_apptainer
#SBATCH --time= 3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1   
#SBATCH --constraint=gpu-80gb

mkdir -p ./graphormer_checkpoints

apptainer exec --nv \
    --bind ./graphormer_checkpoints:/workspace/Graphormer-RT/checkpoints \
    graphormercontainer.sif bash -c "
pip install tensorboardX &&
pip install 'Cython<3.0' &&
cd /workspace/Graphormer-RT/examples/property_prediction/
chmod +x RP.sh  
bash RP.sh
"

exit

