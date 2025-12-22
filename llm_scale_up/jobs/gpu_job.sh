#!/bin/bash -l

# ============================================================================
# Usage: qsub gpu_job.sh <model_size>
#
# Examples:
#   qsub gpu_job.sh tiny
#   qsub gpu_job.sh small
#   qsub gpu_job.sh base
#   qsub gpu_job.sh medium
#
# The job name and output files will be automatically set based on model size
# e.g., MediumQA_1213 -> MediumQA_1213.o<jobid>, MediumQA_1213.e<jobid>
# ============================================================================

# Request 48 hours of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=16:00:00

# Request 32 gigabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=10G

# Request 10 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=10G

# For 1 GPU
#$ -l gpu=1

# Request 8 cores.
#$ -pe smp 8

# Set the working directory to somewhere in your scratch space.
#$ -wd /home/zceccgr/Scratch/freezeLLM

# Parse model size argument (default to 'tiny' if not provided)
MODEL_SIZE=${1:-tiny}

# Map model size to display name for job naming
case "$MODEL_SIZE" in
    tiny)   MODEL_NAME="TinyQA" ;;
    small)  MODEL_NAME="SmallQA" ;;
    base)   MODEL_NAME="BaseQA" ;;
    medium) MODEL_NAME="MediumQA" ;;
    *)
        echo "Error: Invalid model size '$MODEL_SIZE'"
        echo "Valid options: tiny, small, base, medium"
        exit 1
        ;;
esac

# Generate job name with date (MMDD format)
DATE_STR=$(date +%m%d)
JOB_NAME="${MODEL_NAME}_${DATE_STR}"

echo "============================================"
echo "Starting job: $JOB_NAME"
echo "Model size: $MODEL_SIZE"
echo "Date: $(date)"
echo "============================================"

module purge

module load python/miniconda3/24.3.0-0
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate llm-env

nvidia-smi

export PYTHONPATH="/home/zceccgr/Scratch/freezeLLM:$PYTHONPATH"

echo "Starting Python script for $MODEL_NAME model..."
python /home/zceccgr/Scratch/freezeLLM/llm_scale_up/pretrain_wikitext_finetune_qasrl.py \
    --config_path /home/zceccgr/Scratch/freezeLLM/llm_scale_up/config.json \
    --config_name "$MODEL_SIZE"

echo "============================================"
echo "Job completed: $(date)"
echo "============================================"
