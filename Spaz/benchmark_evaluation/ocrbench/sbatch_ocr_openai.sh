#!/bin/bash
#
# Submit OCR Benchmark inference as SLURM batch job on CPU nodes (OpenAI GPT-4o-mini)
# For running ocrbench.py inference tasks with OpenAI models
#

# Submit OCR Benchmark job
echo "🚀 Submitting OCR Benchmark OpenAI job..."
JOB1=$(sbatch --parsable \
  --account=bdpn-delta-cpu \
  --partition=cpu \
  --nodes=1 \
  --cpus-per-task=16 --mem=128G \
  --time=10:00:00 \
  --job-name=ocr-bench-openai \
  --output=logs/ocr/ocr_benchmark_openai_%j.log \
  --wrap="/usr/bin/time -v bash -lc 'source ~/.bashrc && conda activate multi_agent && cd /projects/bdpn/haoqic2/MPU-RL/src/multi-agent/benchmark_evaluation/ocrbench && python infer_ocrbench_openai.py'")

echo "✅ OCR Benchmark OpenAI job submitted: $JOB1"

echo ""
echo "📋 Job Status:"
squeue -j "$JOB1"

echo ""
echo "💡 Monitor with:"
echo "  squeue -j $JOB1"
echo "  squeue -u \$USER"

echo ""
echo "📄 Logs will be in:"
echo "  logs/ocr/ocr_benchmark_openai_${JOB1}.log"

echo ""
echo "📊 Results will be saved to:"
echo "  outputs_ocrbench_openai/"

echo ""
echo "🔍 To check progress:"
echo "  tail -f logs/ocr/ocr_benchmark_openai_${JOB1}.log"
echo ""
echo "📊 To monitor memory usage:"
echo "  sstat -j ${JOB1}.batch --format=MaxRSS,AveRSS,MaxVMSize"
echo ""
echo "💾 To check peak memory after job completes:"
echo "  grep 'Maximum resident set size' logs/ocr/ocr_benchmark_openai_${JOB1}.log"
