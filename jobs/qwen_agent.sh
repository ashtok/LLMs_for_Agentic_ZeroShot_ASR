#!/bin/bash
#SBATCH --job-name=qwen_agent
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/qwen_agent_%j.out
#SBATCH --error=logs/qwen_agent_%j.err

echo "=========================================="
echo "Qwen ASR Agent (Single-GPU)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p logs
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

nvidia-smi

echo ""
echo "Running Qwen Agent (using config defaults)..."
echo ""

# Now uses config defaults - no need to specify model
uv run main/orchestrator.py --max-files 2400 --no-flash-attention

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
