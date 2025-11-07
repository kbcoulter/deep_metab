#!/bin/bash
### NOTE FILEPATH... MOVE ONE LEVEL IN, AS WE ARE NOW WITHIN deep_metab

#SBATCH --account=bgmp
#SBATCH --partition=gpu
#SBATCH --job-name=deep_metab_apptainer_eval
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1   
#SBATCH --constraint=gpu-10gb
#SBATCH --output=evaluate_RP_%j.out.log
#SBATCH --error=evaluate_RP_%j.err.log

apptainer exec --nv \
    --bind ./graphormer_checkpoints:/workspace/Graphormer-RT/checkpoints \
    graphormercontainer.sif bash -c "
    source /opt/conda/bin/activate /opt/conda/envs/graphormer-rt && \
    ./Graphormer-RT/graphormer/evaluate/evaluate_RP.sh
"
