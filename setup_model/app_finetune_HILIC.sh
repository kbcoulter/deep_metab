#!/bin/bash
## TAKE EXTREME CAUTION USING THIS METHOD.
#SBATCH --account=bgmp
#SBATCH --partition=gpu
#SBATCH --job-name=apptainer_transferlearn_HILIC
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1   
#SBATCH --constraint=gpu-10gb
#SBATCH --output=tlearn_HILIC_%j.out.log
#SBATCH --error=tlearn_HILIC_%j.err.log

set -e

# Default
HILIC_DATA_FILE_PATH_DEFAULT="/workspace/Graphormer-RT/my_data/HILIC_ft/DM_Finetune_train.csv"
HILIC_METADATA_PATH_DEFAULT="/workspace/Graphormer-RT/my_data/HILIC_ft/HILIC_metadata.pickle"
BATCH_SIZE_DEFAULT=48
MAX_EPOCH_DEFAULT=100
USER_DATA_DIR_DEFAULT="my_data/HILIC_ft"
CHECKPOINT_DIR_DEFAULT="/workspace/Graphormer-RT/checkpoints_HILIC"
PRETRAINED_MODEL_DEFAULT="${CHECKPOINT_DIR_DEFAULT}/HILIC_WEIGHTS.pt"

# Active
HILIC_DATA_FILE_PATH="${HILIC_DATA_FILE_PATH_DEFAULT}"
HILIC_METADATA_PATH="${HILIC_METADATA_PATH_DEFAULT}"
USER_DATA_DIR="${USER_DATA_DIR_DEFAULT}"
BATCH_SIZE="${BATCH_SIZE_DEFAULT}"
MAX_EPOCH="${MAX_EPOCH_DEFAULT}"
CHECKPOINT_DIR="${CHECKPOINT_DIR_DEFAULT}"
PRETRAINED_MODEL="${PRETRAINED_MODEL_DEFAULT}"

# Args
require_arg() {
  if [[ -z "$2" || "$2" == --* ]]; then
    echo "Error: $1 requires a value"
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-file)
      require_arg "$1" "$2"
      HILIC_DATA_FILE_PATH="$2"
      shift 2
      ;;
    --metadata-file)
      require_arg "$1" "$2"
      HILIC_METADATA_PATH="$2"
      shift 2
      ;;
    --user-data-dir)
      require_arg "$1" "$2"
      USER_DATA_DIR="$2"
      shift 2
      ;;
    --batch-size)
      require_arg "$1" "$2"
      BATCH_SIZE="$2"
      shift 2
      ;;
    --max-epoch)
      require_arg "$1" "$2"
      MAX_EPOCH="$2"
      shift 2
      ;;
    --checkpoint-dir)
      require_arg "$1" "$2"
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --pretrained-model)
      require_arg "$1" "$2"
      PRETRAINED_MODEL="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo ""
      echo "  --data-file <path>           HILIC training CSV (default: ${HILIC_DATA_FILE_PATH_DEFAULT})"
      echo "  --metadata-file <path>       HILIC metadata pickle (default: ${HILIC_METADATA_PATH_DEFAULT})"
      echo "  --user-data-dir <path>       User data directory (default: ${USER_DATA_DIR_DEFAULT})"
      echo "  --batch-size <int>           Batch size (default: ${BATCH_SIZE_DEFAULT})"
      echo "  --max-epoch <int>            Max epochs (default: ${MAX_EPOCH_DEFAULT})"
      echo "  --checkpoint-dir <path>      Checkpoint save directory (default: ${CHECKPOINT_DIR_DEFAULT})"
      echo "  --pretrained-model <path>    Path to pretrained model (default: ${PRETRAINED_MODEL_DEFAULT})"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Export Env
export HILIC_DATA_FILE_PATH
export HILIC_METADATA_PATH

# Finetuning
echo "Starting HILIC finetuning..."
apptainer exec --nv \
  --bind ../Graphormer-RT:/workspace/Graphormer-RT \
  --bind ../graphormer_checkpoints_HILIC:"${CHECKPOINT_DIR}" \
  --bind ../my_data/HILIC_ft:/workspace/Graphormer-RT/my_data/HILIC_ft \
  ../graphormercontainer.sif bash -lc "
source /opt/conda/bin/activate /opt/conda/envs/graphormer-rt
export PYTHONPATH=/workspace/Graphormer-RT/graphormer/modules:\$PYTHONPATH
cd /workspace/Graphormer-RT
CUDA_VISIBLE_DEVICES=0 fairseq-train \
    --user-dir graphormer \
    --batch-size ${BATCH_SIZE} \
    --num-workers 20 \
    --ddp-backend legacy_ddp \
    --seed 23 \
    --user-data-dir ${USER_DATA_DIR} \
    --dataset-name DM_Finetune \
    --task graph_prediction_with_flag \
    --criterion rmse_HILIC \
    --arch graphormer_HILIC \
    --num-classes 1 \
    --attention-dropout 0.03 --act-dropout 0.2 --dropout 0.02 \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 1.0 --weight-decay 0.1 \
    --lr-scheduler polynomial_decay --power 1 --warmup-updates 23 --total-num-update 156 \
    --lr 5e-6 \
    --fp16 \
    --encoder-layers 8 \
    --encoder-embed-dim 512 \
    --encoder-ffn-embed-dim 512 \
    --encoder-attention-heads 64 \
    --mlp-layers 5 \
    --max-epoch ${MAX_EPOCH} \
    --no-epoch-checkpoints \
    --freeze-level 0 \
    --save-dir ${CHECKPOINT_DIR} \
    --pretrained-model-name ${PRETRAINED_MODEL} \
    --finetune-from-model ${PRETRAINED_MODEL}
"