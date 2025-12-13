#!/bin/bash

# ============================================================================
# Submit a training job with automatic job naming
#
# Usage: ./submit_job.sh <model_size>
#
# Examples:
#   ./submit_job.sh tiny
#   ./submit_job.sh small
#   ./submit_job.sh base
#   ./submit_job.sh medium
#
# This will submit a job named e.g., MediumQA_1213 with output files:
#   MediumQA_1213.o<jobid>
#   MediumQA_1213.e<jobid>
# ============================================================================

set -e

# Check if model size argument is provided
if [ -z "$1" ]; then
    echo "Error: No model size specified"
    echo "Usage: ./submit_job.sh <model_size>"
    echo "Options: tiny, small, base, medium"
    exit 1
fi

MODEL_SIZE="$1"

# Map model size to display name
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

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting job: $JOB_NAME"
echo "Model size: $MODEL_SIZE"

# Submit the job with the generated name
qsub -N "$JOB_NAME" "$SCRIPT_DIR/gpu_job.sh" "$MODEL_SIZE"

echo "Job submitted successfully!"