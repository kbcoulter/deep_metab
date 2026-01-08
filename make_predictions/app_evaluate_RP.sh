#!/bin/bash
### NOTE: RUN THIS FROM deep_metab DIR

#SBATCH --account=bgmp
#SBATCH --partition=gpu
#SBATCH --job-name=apptainer_eval_RP
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1   
#SBATCH --constraint=gpu-10gb
#SBATCH --output=evaluate_RP_%j.out.log
#SBATCH --error=evaluate_RP_%j.err.log

HOST_DATA_DIR="/projects/bgmp/shared/groups/2025/deepmetab/kcoulter/deep_metab/my_data/sample_data_0001"
# SET ABSOLUTE PATH CONTAINING 1 .csv FOR RT AND 1 .pkl FOR CHROMATOGRAPHY CONDITIONS

DATA_DIR="${SLURM_JOB_ID}.csv"
CONTAINER_DATA_FILE=$(find "$HOST_DATA_DIR" -maxdepth 1 -name "*.csv" -print -quit) # <-- Retention time data
CONTAINER_METADATA_FILE=$(find "$HOST_DATA_DIR" -maxdepth 1 -name "*.pickle" -print -quit) # <-- Chromatography sample conditions
apptainer exec --nv \
    --bind ./graphormer_checkpoints_RP:/workspace/Graphormer-RT/checkpoints_RP \
    --bind ${HOST_DATA_DIR}:/data \
    graphormercontainer.sif bash -c "
    source /opt/conda/bin/activate /opt/conda/envs/graphormer-rt && \
    export RP_DATA_FILE_PATH=\"${CONTAINER_DATA_FILE}\"
    export RP_METADATA_PATH=\"${CONTAINER_METADATA_FILE}\"
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
        --save-path '../../../predictions_RP/$DATA_DIR' \
        --save-dir '/workspace/Graphormer-RT/checkpoints_RP/' \
        --split train
"
