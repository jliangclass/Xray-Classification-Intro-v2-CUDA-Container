#!/usr/bin/env bash
#SBATCH --job-name=xray14-convnext-adamw-best
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1

# Examples for custom data directory
# ./apptainer-run-xray14-convnext-base-best.sh dataset.dataset_root=$XRAY_DATASET
# ./apptainer-run-xray14-convnext-base-best.sh debug=True dataset.dataset_root=$XRAY_DATASET train.batch_size=128

# for Nvidia a100, use batch size = 128
# for Nvidia v100, use batch size = 64

DATASET="chestxray14"
MODEL="convnext_base"
OPTIMIZER="adamw"
LR="2e-4"
BATCH_SIZE=64

OUTPUT_DIR="$SCRATCH/outputs-Xray-Classification-Intro-v2/$CLUSTER"
mkdir -p $OUTPUT_DIR

APPTAINER_CMD=(./cuda-apptainer.sh exec)

# Check if first argument is a number of GPUs
if [[ $1 =~ ^[0-9]+$ ]]; then
    NUM_GPUS=$1
    shift
else
    NUM_GPUS=1
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    FREE_PORT=$("${APPTAINER_CMD[@]}" python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    "${APPTAINER_CMD[@]}" torchrun --nproc_per_node=$NUM_GPUS --master_port=$FREE_PORT src/train_v1_classic.py dataset=$DATASET model=$MODEL train.batch_size=$BATCH_SIZE train.optimizer=$OPTIMIZER train.lr=$LR scheduler=reduce_lr_on_plateau hydra.run.dir=$OUTPUT_DIR "$@"
else
    "${APPTAINER_CMD[@]}" python src/train_v1_classic.py dataset=$DATASET model=$MODEL train.batch_size=$BATCH_SIZE train.optimizer=$OPTIMIZER train.lr=$LR scheduler=reduce_lr_on_plateau hydra.run.dir=$OUTPUT_DIR "$@"
fi
