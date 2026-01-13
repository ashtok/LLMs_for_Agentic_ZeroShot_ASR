#!/bin/bash
#SBATCH --job-name=asr_agent          # was qwen_agent
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/asr_agent_%j.out   # was qwen_agent_%j.out
#SBATCH --error=logs/asr_agent_%j.err    # was qwen_agent_%j.err

echo "=========================================="
echo "ASR Judge Agent (Single-GPU)"            # generic label
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p logs
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

nvidia-smi

echo ""
echo "Running ASR Agent (using config defaults)..."
echo ""

# Orchestrator will pick Qwen / Llama / Gemma via config.CURRENT_BACKBONE
uv run main/orchestrator.py --max-files 2400 --no-flash-attention

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
