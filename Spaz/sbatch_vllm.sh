#!/bin/bash
#
# Submit vLLM servers as SLURM batch jobs
# More reliable than salloc for automation
#
# Usage: ./sbatch_vllm.sh [vision_port] [text_port]
# Example: ./sbatch_vllm.sh 8000 8001  (default)
#          ./sbatch_vllm.sh 8002 8003  (second instance)
#          ./sbatch_vllm.sh 8004 8005  (third instance)
#

# Parse port arguments
VISION_PORT=${1:-8000}  # Default to 8000 if not provided
TEXT_PORT=${2:-8001}    # Default to 8001 if not provided

echo "🔧 Using ports: Vision=${VISION_PORT}, Text=${TEXT_PORT}"

# Submit Vision Model job
echo "🚀 Submitting Vision Model job..."
JOB1=$(sbatch --parsable \
  --account=bdpn-delta-gpu \
  --partition=gpuA40x4 \
  --nodes=1 --gpus-per-node=1 \
  --cpus-per-task=8 --mem=30G \
  --time=48:00:00 \
  --job-name=vllm-vision-${VISION_PORT} \
  --output=logs/vllm/vllm_vision_${VISION_PORT}_%j.log \
  --wrap="source ~/.bashrc && conda activate vllm_infer && python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --port ${VISION_PORT} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --disable-custom-all-reduce \
    --dtype float16 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 8192 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --chat-template /projects/bdpn/wzhang29/MPU-RL/chat_template.jinja")

echo "✅ Vision Model job submitted: $JOB1"

# Submit Text Model job
echo "🚀 Submitting Text Model job..."
JOB2=$(sbatch --parsable \
  --account=bdpn-delta-gpu \
  --partition=gpuA40x4 \
  --nodes=1 --gpus-per-node=1 \
  --cpus-per-task=8 --mem=30G \
  --time=48:00:00 \
  --job-name=vllm-text-${TEXT_PORT} \
  --output=logs/vllm/vllm_text_${TEXT_PORT}_%j.log \
  --wrap="source ~/.bashrc && conda activate vllm_infer && python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --port ${TEXT_PORT} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --disable-custom-all-reduce \
    --dtype float16 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 8192 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes")

echo "✅ Text Model job submitted: $JOB2"

echo ""
echo "📋 Job Status:"
squeue -j "$JOB1,$JOB2"

echo ""
echo "💡 Monitor with:"
echo "  squeue -j $JOB1,$JOB2"
echo "  # When running, set up port forwarding:"
echo "  python restart_vllm_with_tools.py"

echo ""
echo "📄 Logs will be in:"
echo "  logs/vllm/vllm_vision_${VISION_PORT}_${JOB1}.log"
echo "  logs/vllm/vllm_text_${TEXT_PORT}_${JOB2}.log"