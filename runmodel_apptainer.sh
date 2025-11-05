#!/bin/bash

#SBATCH --account=bgmp
#SBATCH --partition=gpu
#SBATCH --job-name=deep_metab_apptainer
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1   
#SBATCH --constraint=gpu-10gb
#SBATCH --output=metab_%j.out.log
#SBATCH --error=metab_%j.err.log

module load nvidia/cuda

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


