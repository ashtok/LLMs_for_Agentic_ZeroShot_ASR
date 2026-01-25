#!/bin/bash
#SBATCH --job-name=asr_agent_optimized
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16              # Increased for 4 parallel threads
#SBATCH --mem=64G                       # Increased for parallel models + 30B LLM
#SBATCH --time=24:00:00
#SBATCH --output=logs/asr_agent_%j.out
#SBATCH --error=logs/asr_agent_%j.err

echo "=========================================="
echo "ASR Judge Agent (Optimized Single-GPU)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Optimizations:"
echo "  - Parallel baseline processing (4 threads)"
echo "  - Flash Attention 2 enabled"
echo "  - Increased batch size"
echo "  - 16 CPU cores for threading"
echo "=========================================="

mkdir -p logs
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

# Set threading environment variables for better performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# One-time snapshot
nvidia-smi

# Start background GPU utilization logging
LOG_DIR=logs
mkdir -p "$LOG_DIR"

nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv \
  -l 1 \
  -f "$LOG_DIR/gpu_usage_${SLURM_JOB_ID}.csv" &
GPU_LOG_PID=$!

echo ""
echo "Running ASR Agent with optimizations..."
echo ""

# Run with optimized settings:
# - Remove --no-flash-attention to enable Flash Attention 2
# - Optionally add --load-8bit if you need more memory headroom
uv run main/orchestrator.py --max-files 750

JOB_STATUS=$?

# Stop GPU logger
kill $GPU_LOG_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo "Job finished at: $(date) (status $JOB_STATUS)"
echo "GPU log: $LOG_DIR/gpu_usage_${SLURM_JOB_ID}.csv"
echo "=========================================="
