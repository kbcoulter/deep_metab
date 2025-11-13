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
#SBATCH --output=evaluate_HILIC_%j.out.log
#SBATCH --error=evaluate_HILIC_%j.err.log

apptainer exec --nv \
    --bind ./graphormer_checkpoints_HILIC:/workspace/Graphormer-RT/checkpoints_HILIC \
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
        --save-path '../../../predictions_HILIC/HILIC_preds.csv' \
        --save-dir '/workspace/Graphormer-RT/checkpoints_HILIC/' \
        --split train
"
