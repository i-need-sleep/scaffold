#!/bin/bash
source activate a2a
export CUDA_VISIBLE_DEVICES=0


logfile="logs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$logfile") 2>&1
echo "Logging output to $logfile"
echo "Starting script at $(date)"

cd /home/willhuang/projects/audio2audio/code
python -u main.py \
    --name placeholder \
    --experiment_group placeholder \
    --task placeholder \
    --mode predict_dev \
    --lr 3e-3 \
    --max_n_epoch -1

echo "Script ended at $(date)"