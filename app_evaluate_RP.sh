#!/bin/bash
### NOTE: RUN THIS FROM deep_metab

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

# Folder name of Graphormer-RT predictions initialized to preds_<sbatch job #>
PREDS_DIR="preds_${SLURM_JOB_ID}"

# Create the directory on the host *before* running the container
mkdir -p "$PREDS_DIR"

# Run evaluate.py in Graphormer-RT under our container
apptainer exec --nv \
    --bind ./graphormer_checkpoints_RP:/workspace/Graphormer-RT/checkpoints_RP \
    graphormercontainer.sif bash -c "
    source /opt/conda/bin/activate /opt/conda/envs/graphormer-rt && \
    cd ./Graphormer-RT/graphormer/evaluate/ && \
    python evaluate.py \
        --user-dir ../../graphormer \
        --num-workers 32 \
        --ddp-backend=legacy_ddp \
        --user-data-dir rp_test \
        --dataset-name RT_test \
        --task graph_prediction \
        --criterion rmse \
        --arch graphormer_base \
        --encoder-layers 8 \
        --encoder-embed-dim  512 \
        --encoder-ffn-embed-dim 512 \
        --encoder-attention-heads 64 \
        --freeze-level -4 \
        --mlp-layers 5 \
        --batch-size 64 \
        --num-classes 1 \
        --save-path '../../../$PREDS_DIR/RP_preds.csv' \
        --save-dir '/workspace/Graphormer-RT/checkpoints_RP/' \
        --split train
"
