#!/bin/bash

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

set -e

# Default
HOST_DATA_DIR_DEFAULT="../my_data/sample_data_0001"
CHECKPOINT_DIR_DEFAULT="./graphormer_checkpoints_RP"
SAVE_PATH_DEFAULT="./predictions_RP"

# Active 
HOST_DATA_DIR="${HOST_DATA_DIR_DEFAULT}"
CHECKPOINT_DIR="${CHECKPOINT_DIR_DEFAULT}"
SAVE_PATH="${SAVE_PATH_DEFAULT}"

# Args
require_arg() {
  if [[ -z "$2" || "$2" == --* ]]; then
    echo "Error: $1 requires a value"
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host-data-dir)
      require_arg "$1" "$2"
      HOST_DATA_DIR="$2"
      shift 2
      ;;
    --checkpoint-dir)
      require_arg "$1" "$2"
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --save-path)
      require_arg "$1" "$2"
      SAVE_PATH="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo ""
      echo "  --host-data-dir <path>      Directory containing 1 .csv + 1 .pkl (default: ${HOST_DATA_DIR_DEFAULT})"
      echo "  --checkpoint-dir <path>     Checkpoint directory (default: ${CHECKPOINT_DIR_DEFAULT})"
      echo "  --save-path <path>          Path to save predictions (default: ${SAVE_PATH_DEFAULT})"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Collect Data
DATA_CSV=$(find "$HOST_DATA_DIR" -maxdepth 1 -name "*.csv" -print -quit)
DATA_PKL=$(find "$HOST_DATA_DIR" -maxdepth 1 -name "*.pickle" -print -quit)
DATA_DIR="${SLURM_JOB_ID}.csv"

# Run App
apptainer exec --nv \
    --bind "${CHECKPOINT_DIR}":/workspace/Graphormer-RT/checkpoints_RP \
    --bind "${HOST_DATA_DIR}":/data \
    graphormercontainer.sif bash -c "
    source /opt/conda/bin/activate /opt/conda/envs/graphormer-rt && \
    export RP_DATA_FILE_PATH=\"${DATA_CSV}\" && \
    export RP_METADATA_PATH=\"${DATA_PKL}\" && \
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
        --encoder-embed-dim 512 \
        --encoder-ffn-embed-dim 512 \
        --encoder-attention-heads 64 \
        --freeze-level -4 \
        --mlp-layers 5 \
        --batch-size 64 \
        --num-classes 1 \
        --save-path '${SAVE_PATH}/$DATA_DIR' \
        --save-dir '${CHECKPOINT_DIR}' \
        --split train
"