#!/bin/bash
#SBATCH --job-name araieval-evaluation
#SBATCH --partition A100-40GB
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --time=03-00:00:00

# arg1 = input file
# arg2 = model checkpoint file


srun --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
    --container-workdir="`pwd`" \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
    --task-prolog="`pwd`/install.sh" \
    python task1/src/generate.py task1/data/araieval24_task1_train.jsonl checkpoints/exp1/model_best.pt
