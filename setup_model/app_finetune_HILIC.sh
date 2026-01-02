#!/bin/bash
## TAKE EXTREME CAUTION TO AVOID OVERFITTING! THIS METHOD IS LESS RE-PRODUCIBLE
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

apptainer exec --nv \
    --bind ./graphormer_checkpoints_HILIC:/workspace/Graphormer-RT/checkpoints_HILIC \
    graphormercontainer.sif bash -c "
    source /opt/conda/bin/activate /opt/conda/envs/graphormer-rt && \
    cd ../ && \
	CUDA_VISIBLE_DEVICES=0 fairseq-train \
	--user-dir ../../graphormer \
	--batch-size 64 \
	--batch-size 64 \
	--num-workers 20 \
	--ddp-backend=legacy_ddp \
	--seed 23 \
	--user-data-dir HILIC_train \
	--dataset-name RT_Library \
	--task graph_prediction_with_flag \
	--criterion rmse_HILIC \
	--arch graphormer_HILIC \
	--num-classes 1 \
	--attention-dropout 0.15 --act-dropout 0.10 --dropout 0.10 \
	--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.01 \
	--lr-scheduler polynomial_decay --power 1 --warmup-updates 10 --total-num-update  650 \
	--lr 5e-5 \
	--fp16 \
    --encoder-layers 8 \
    --encoder-embed-dim 512 \
    --encoder-ffn-embed-dim 512 \
    --encoder-attention-heads 64 \
    --mlp-layers 5 \
	--max-epoch 250 \
	--no-epoch-checkpoints \
	--freeze-level 0 \
	--save-dir ../../graphormer_checkpoints_HILIC \
	--pretrained-model-name ../../models/HILIC/55.pt\
	--finetune-from-model ../../models/HILIC/55.pt \
