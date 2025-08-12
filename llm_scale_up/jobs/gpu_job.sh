#!/bin/bash -l

# Request 1 hour of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=16:00:00

# Request 256 gigabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=8G

# Request 10 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=10G

# For 1 GPU
#$ -l gpu=1

# Request 10 cores.
#$ -pe smp 8

# Set the name of the job.
#$ -N llmfreeze_aug12_1

# Set the working directory to somewhere in your scratch space.
#  This is a necessary step as compute nodes cannot write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/zceccgr/Scratch/freezeLLM

module purge


module load cuda/12.2.2/gnu-10.2.0
module load python/miniconda3/24.3.0-0

source $UCL_CONDA_PATH/etc/profile.d/conda.sh

conda activate llm_env



nvidia-smi

# Activate python environment
source /home/zccecgr/llmenv4/bin/activate

# Your work should be done in $TMPDIR
cd $TMPDIR

python /home/zccecgr/Scratch/freezeLLM/llm_scale_up/pretrain_wikitext_finetune_qasrl.py
