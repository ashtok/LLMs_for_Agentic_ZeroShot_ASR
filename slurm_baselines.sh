#!/bin/bash
#SBATCH --job-name=asr_baselines
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/asr_baselines_%j.out
#SBATCH --error=logs/asr_baselines_%j.err

echo "=========================================="
echo "ASR Baseline Evaluations"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Set library path for fairseq2/OmniASR
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Display system info
echo ""
echo "System Information:"
echo "-------------------"
nvidia-smi
echo ""
echo "Python environment:"
uv run python --version
echo ""
echo "Conda environment:"
echo $CONDA_PREFIX
echo ""

echo "=========================================="
echo "Starting baseline evaluations..."
echo "=========================================="

# Run all baselines
uv run main/compare_baselines.py

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Job completed successfully!"
else
    echo "Job failed with exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
