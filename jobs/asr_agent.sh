#!/bin/bash
#SBATCH --job-name=asr_agent
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/asr_agent_%j.out
#SBATCH --error=logs/asr_agent_%j.err

echo "=========================================="
echo "ASR Judge Agent (Single-GPU)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p logs
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export FAIRSEQ2_NO_LIBSNDFILE=1  # ✅ FIX: Skip libsndfile requirement for OmniASR

echo ""
echo "GPU Info:"
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
echo "Running ASR Agent (Qwen/Llama/Gemma via config.CURRENT_BACKBONE)..."
echo "Max files: 750, No Flash Attention"
echo ""

# Run orchestrator with error handling
uv run main/orchestrator.py --max-files 750 --no-flash-attention
JOB_STATUS=$?

# Stop GPU logger
if [ ! -z "$GPU_LOG_PID" ]; then
    kill $GPU_LOG_PID 2>/dev/null || true
    wait $GPU_LOG_PID 2>/dev/null
fi

echo ""
echo "=========================================="
echo "Job finished at: $(date) (status: $JOB_STATUS)"
echo "GPU log: $LOG_DIR/gpu_usage_${SLURM_JOB_ID}.csv"
echo "=========================================="

exit $JOB_STATUS
