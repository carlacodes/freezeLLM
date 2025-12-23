#!/bin/bash -l

# ============================================================================
# Usage: qsub download_datasets.sh
#
# Pre-downloads all HuggingFace datasets to Scratch before training.
# Run this ONCE before submitting training jobs.
# ============================================================================

# Request 2 hours of wallclock time
#$ -l h_rt=2:00:00

# Request 16 gigabyte of RAM
#$ -l mem=16G

# Request 10 gigabyte of TMPDIR space
#$ -l tmpfs=10G

# Set the name of the job
#$ -N download_datasets

# Set the working directory
#$ -wd /home/zceccgr/Scratch/freezeLLM

echo "============================================"
echo "Starting dataset download job"
echo "Date: $(date)"
echo "============================================"

module purge

module load python/miniconda3/24.3.0-0
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate llm-env

export PYTHONPATH="/home/zceccgr/Scratch/freezeLLM:$PYTHONPATH"

echo "Downloading datasets to /home/zceccgr/Scratch/huggingface_cache ..."
python /home/zceccgr/Scratch/freezeLLM/llm_scale_up/download_datasets.py

echo "============================================"
echo "Download completed: $(date)"
echo "============================================"