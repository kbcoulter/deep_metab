#!/bin/bash
### BROKEN! 
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

mkdir -p ./graphormer_checkpoints

apptainer exec --nv \
    --bind ./graphormer_checkpoints:/workspace/Graphormer-RT/checkpoints \
    graphormercontainer.sif bash -c "
source /opt/conda/bin/activate /opt/conda/envs/graphormer-rt && \
cd /workspace/Graphormer-RT/examples/property_prediction/ && \
bash RP.sh
"


