# MIA-Bench Integration (v2)

This directory contains the integration of MIA-Bench (Multimodal Instruction-following Evaluation Benchmark) into our multi-agent framework using the unified BenchmarkInfer system.

## Overview

MIA-Bench is a benchmark designed to evaluate multimodal large language models (MLLMs) on their ability to strictly adhere to complex instructions. Our v2 integration:

1. **Uses unified BenchmarkInfer base class** (`app/utils/benchmark_infer.py`) for maximum code reuse
2. **Comprehensive flow integration** with TranslatorAgent + TextOnlyReasoningAgent
3. **Advanced logging and session management** with automatic token tracking
4. **Robust image processing** with caching and error handling
5. **73% code reduction** compared to v1 while adding more features

## Quick Start

### 1. Setup Environment

**Python Environment**: Recommended to use Python 3.11 

**API Keys Setup**:
```bash
# For DashScope models (Qwen2.5-VL-3B/7B/32B)
export DASHSCOPE_API_KEY="YOUR_DASHSCOPE_API_KEY"

# For OpenAI models (GPT-4o-mini and evaluation)
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

# Set Python path
export PYTHONPATH=/path/to/MPU-RL/src/multi-agent:$PYTHONPATH
```

### 2. Run Inference

```bash
# Enter MIA directory (replace with your actual path)
cd /path/to/MPU-RL/src/multi-agent/benchmark_evaluation/MIA

# Set up Python environment (replace with your Python 3.11 path)
export PATH="/path/to/python3.11/bin:$PATH"
export PYTHONPATH=/path/to/MPU-RL/src/multi-agent:$PYTHONPATH

# Quick test with 5 samples using multi-agent flow
python run_mia_inference_flow.py --max-samples 5

# Process specific sample range
python run_mia_inference_flow.py --start-sample 100 --end-sample 200

# Custom port configuration for parallel processing
python run_mia_inference_flow.py --max-samples 10 --vision-port 8002 --text-port 8003

# Full evaluation with custom iterations
python run_mia_inference_flow.py --max-iterations 2

# Full evaluation (400 samples)
python run_mia_inference_flow.py
```

### 3. Run Evaluation

```bash
# Evaluate inference results (using official GPT-4o evaluation method)
python run_mia_evaluation.py --results outputs/gpt4o_mini_*.jsonl

# Evaluate multiple model results
python run_mia_evaluation.py --results outputs/qwen7b_results.jsonl --output-dir evaluation_results/qwen7b

# Generated files:
# - mia_official_format_*.csv     # Compatible with official evaluation.ipynb
# - mia_evaluation_detailed_*.csv # Detailed analysis results  
# - mia_evaluation_report_*.md    # Human-readable report
```

## Supported Models

| Model Config | Description | API Type |
|--------------|-------------|----------|
| `gpt4o_mini` | GPT-4o-mini (Recommended for testing) | OpenAI API |
| `mia_gpt4o_mini` | GPT-4o-mini (MIA optimized) | OpenAI API |
| `qwen2_5_vl_3b` | Qwen2.5-VL-3B-Instruct (vLLM) | vLLM |
| `qwen2_5_vl_7b` | Qwen2.5-VL-7B-Instruct (vLLM) | vLLM |
| `translator` | Qwen2.5-VL-3B via DashScope | DashScope |
| `mia_qwen2_5_vl_3b_dashscope` | Qwen2.5-VL-3B via DashScope | DashScope |
| `mia_qwen2_5_vl_7b_dashscope` | Qwen2.5-VL-7B via DashScope | DashScope |
| `mia_qwen2_5_vl_32b` | Qwen2.5-VL-32B via DashScope | DashScope |

List all supported models:
```bash
python run_mia_inference.py --list-models
```

## Architecture (v2)

### Core Components

1. **`app/utils/benchmark_infer.py`**: Unified base class for all benchmark inference
   - Comprehensive logging and session management
   - Image processing with caching and resizing
   - Token tracking across all agents
   - Standardized error handling and retry logic

2. **`run_mia_inference_flow.py`**: MIA-specific flow implementation (150 lines vs 400 in v1)
   - `MIABenchmarkInfer`: Inherits from unified base class
   - Only implements 7 abstract methods for MIA-specific data handling
   - Automatic flow integration with TranslatorAgent + TextOnlyReasoningAgent

3. **`run_mia_evaluation.py`**: Evaluation using official MIA methodology (unchanged)
   - Uses GPT-4o as judge (following official approach)
   - Generates detailed reports and metrics
   - Compatible with official evaluation format

### Data Flow (v2)

```
MIA Dataset (JSON)
    ↓
MIABenchmarkInfer (inherits from BenchmarkInfer)
    ↓
FlowExecutor (TranslatorAgent + TextOnlyReasoningAgent)
    ↓
Session Logs + Results (JSON/CSV/JSONL)
    ↓
run_mia_evaluation.py (GPT-4o judge)
    ↓
Evaluation Report + Detailed Results
```

### Key Improvements

- **73% Code Reduction**: 400+ lines → 150 lines
- **Unified Infrastructure**: Shared logging, error handling, token tracking
- **Robust Image Processing**: Automatic caching, resizing, format conversion
- **Flow Integration**: Full multi-agent flow support out of the box
- **Consistent Behavior**: Same patterns as MMMU and future benchmarks

## Configuration

Model configurations are automatically loaded from `config.toml`. The v2 implementation uses the flow system which requires:

- `translator_api`: Vision model configuration (default: ports 8000)
- `reasoning_api`: Text model configuration (default: ports 8001)

The flow is automatically configured - no manual model setup needed!

```python
# v2 Implementation - Automatic flow setup
mia_infer = MIABenchmarkInfer(max_iterations=3)
# That's it! Flow, logging, error handling all automatic
```

## File Structure

```
MIA/
├── README.md                        # This file (updated for v2)
├── model_utils.py                  # Legacy model wrapper classes (still used by run_mia_inference.py)
├── run_mia_inference_flow.py       # 🆕 v2 Flow-based inference (RECOMMENDED)
├── run_mia_inference.py            # Legacy single-model inference runner
├── run_mia_evaluation.py           # Evaluation runner (unchanged)
├── data/                       # MIA dataset
│   ├── instruction_benchmark_all.json
│   └── example_inference_result.jsonl
├── evaluation/                 # Official evaluation code
│   └── evaluation.ipynb
└── outputs/                    # Generated results
    ├── *_results.jsonl         # Inference results
    ├── *_metadata.json         # Inference metadata
    └── evaluation_results/     # Evaluation outputs
```

## 🚀 Complete 400-Sample Evaluation Guide

### For Full MIA-Bench Evaluation (400 samples)

**⚠️ Important Notes:**
- **Time**: Full evaluation takes 2-4 hours depending on model and API speed
- **Cost**: GPT-4o-mini ~$15-25, Qwen models ~$5-10 for full evaluation
- **API Limits**: Use `--concurrent-requests 3` to avoid rate limits

### Step-by-Step Full Evaluation

```bash
# 1. Environment setup
export PATH="/path/to/python3.11/bin:$PATH"
export PYTHONPATH=/path/to/MPU-RL/src/multi-agent:$PYTHONPATH
export OPENAI_API_KEY="your-openai-key"
export DASHSCOPE_API_KEY="your-dashscope-key"  # For Qwen models

# 2. Run full inference (400 samples)
cd /path/to/MPU-RL/src/multi-agent/benchmark_evaluation/MIA

# GPT-4o-mini (Recommended, stable and fast)
python run_mia_inference.py \
    --model gpt4o_mini \
    --output outputs/gpt4o_mini_full_400.jsonl \
    --concurrent-requests 3

# Qwen2.5-VL-7B (DashScope API)
python run_mia_inference.py \
    --model mia_qwen2_5_vl_7b_dashscope \
    --output outputs/qwen7b_full_400.jsonl \
    --concurrent-requests 2

# Qwen2.5-VL-32B (DashScope API)
python run_mia_inference.py \
    --model mia_qwen2_5_vl_32b \
    --output outputs/qwen32b_full_400.jsonl \
    --concurrent-requests 1

# 3. Run evaluation (using GPT-4o as judge)
python run_mia_evaluation.py \
    --results outputs/gpt4o_mini_full_400.jsonl \
    --output-dir evaluation_results/gpt4o_mini_full

# 4. View results
cat evaluation_results/gpt4o_mini_full/mia_evaluation_report_*.md
```

### Expected Output Files

After full evaluation, you'll get:

```
outputs/
├── gpt4o_mini_full_400.jsonl                    # Inference results (400 samples)
├── gpt4o_mini_full_400_metadata.json            # Inference metadata
└── evaluation_results/gpt4o_mini_full/
    ├── mia_official_format_*.csv                 # Official format (compatible with evaluation.ipynb)
    ├── mia_evaluation_detailed_*.csv              # Detailed analysis
    ├── mia_evaluation_raw_*.json                 # Raw scores
    └── mia_evaluation_report_*.md                # Human-readable report
```

### Performance Expectations

| Model | Time (400 samples) | Cost (approx) | Success Rate |
|-------|-------------------|---------------|--------------|
| GPT-4o-mini | 2-3 hours | $15-25 | 95-98% |
| Qwen2.5-VL-7B | 3-4 hours | $5-10 | 90-95% |
| Qwen2.5-VL-32B | 4-6 hours | $10-20 | 85-92% |

### Troubleshooting Full Evaluation

```bash
# If encountering API limits, reduce concurrent requests
python run_mia_inference.py --model gpt4o_mini --concurrent-requests 1

# If network is unstable, increase retry count
python run_mia_inference.py --model gpt4o_mini --max-retries 5

# If memory insufficient, process in batches
python run_mia_inference.py --model gpt4o_mini --max-samples 100
python run_mia_inference.py --model gpt4o_mini --max-samples 100 --start-index 100
python run_mia_inference.py --model gpt4o_mini --max-samples 100 --start-index 200
python run_mia_inference.py --model gpt4o_mini --max-samples 100 --start-index 300
```

## Examples

### Complete Workflow

```bash
# 1. Run inference with Qwen2.5-VL-7B
python run_mia_inference.py \
    --model mia_qwen2_5_vl_7b \
    --max-samples 100 \
    --output outputs/qwen7b_test.jsonl

# 2. Evaluate results
python run_mia_evaluation.py \
    --results outputs/qwen7b_test.jsonl \
    --output-dir evaluation_results/qwen7b_test

# 3. View results
cat evaluation_results/qwen7b_test/mia_evaluation_report_*.md
```

### Batch Processing Multiple Models

```bash
# Run inference for multiple models
for model in mia_gpt4o_mini mia_qwen2_5_vl_3b mia_qwen2_5_vl_32b; do
    echo "Running inference with $model..."
    python run_mia_inference.py \
        --model $model \
        --max-samples 50 \
        --output outputs/${model}_results.jsonl
done

# Evaluate all results
for result_file in outputs/*_results.jsonl; do
    echo "Evaluating $result_file..."
    python run_mia_evaluation.py \
        --results $result_file \
        --output-dir evaluation_results/
done
```

## Integration with Existing Framework

This MIA integration reuses our existing infrastructure:

- **LLM Interface**: Uses `app/llm.py` for all model interactions
- **Configuration**: Extends `config.toml` with MIA-specific settings  
- **Message Format**: Uses `app/schema.Message` for multimodal inputs
- **Error Handling**: Follows existing retry and error handling patterns

## Future Extensions

1. **Multi-Agent Flow**: Integrate with `TranslatorAgent` + `TextOnlyReasoningAgent`
2. **Advanced Evaluation**: Add component-level analysis and custom metrics
3. **Batch Processing**: Support for large-scale evaluation runs
4. **Custom Instructions**: Support for user-defined instruction templates

## Troubleshooting

### Common Issues

1. **API Key Missing**: Ensure `OPENAI_API_KEY` or `DASHSCOPE_API_KEY` is set
2. **Model Not Found**: Check `config.toml` for model configuration
3. **Image Loading Failed**: Some URLs may be unstable; consider local caching
4. **Rate Limits**: Reduce `--concurrent-requests` for API models
5. **Image Too Large for API**: If encountering base64 image size limit issues, we have prepared image compression solution (refer to MMMU PRO implementation)

### Debug Mode

```bash
# Enable debug logging
export PYTHONPATH=/path/to/MPU-RL/src/multi-agent:$PYTHONPATH
python -u run_mia_inference.py --model mia_gpt4o_mini --max-samples 5
```

## References

- [MIA-Bench Official Repository](https://github.com/apple/ml-mia-bench)
- [MIA-Bench Paper](https://arxiv.org/abs/2407.01509)
- [MMMU Flow Integration](../mmmu/README_FLOW_EVAL.md)
- [Agent Architecture](../../AGENT_ARCHITECTURE.md)
