# MMMU PRO Single Model Evaluation

This module provides single-model evaluation capabilities for the [MMMU PRO benchmark](https://huggingface.co/datasets/MMMU/MMMU_Pro), following the official implementation structure.

## Overview

MMMU PRO is a more robust version of the MMMU benchmark with:
- **Standard (4 options)**: Traditional 4-choice questions
- **Standard (10 options)**: Increased difficulty with 10 choices
- **Vision**: Questions embedded in images, requiring visual-text integration

**Important**: By default, this implementation evaluates on **validation cases only** (577 samples per subset), which is the standard practice for MMMU PRO evaluation.

## Supported Models

### Qwen Models (via DashScope API)
- `qwen2.5-vl-3b` - Qwen2.5-VL-3B-Instruct
- `qwen2.5-vl-7b` - Qwen2.5-VL-7B-Instruct  
- `qwen2.5-vl-32b` - Qwen2.5-VL-32B-Instruct

### GPT Models (via OpenAI API)
- `gpt-4o-mini` or `gpt4o-mini` - GPT-4o-mini

## Setup

### 1. Environment Variables

**For Qwen models (DashScope):**
```bash
export DASHSCOPE_API_KEY="your_dashscope_api_key"
export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
```

**For GPT models (OpenAI):**
```bash
export OPENAI_API_KEY="your_openai_api_key"
```

### 2. Install Dependencies

```bash
cd src/multi-agent/benchmark_evaluation/mmmu_pro
pip install -r requirements.txt
```

## Quick Start

### Step 1: Navigate to MMMU PRO Directory
```bash
cd src/multi-agent/benchmark_evaluation/mmmu_pro
```

### Step 2: Set Environment Variables

**For Qwen models (DashScope):**
```bash
export DASHSCOPE_API_KEY="your_dashscope_api_key"
export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
```

**For GPT models (OpenAI):**
```bash
export OPENAI_API_KEY="your_openai_api_key"
```

### Step 3: Run Your First Test

**Test GPT-4o-mini on Standard (4 options) with 4 samples:**
```bash
python infer/infer_gpt.py gpt-4o-mini cot "standard (4 options)" --max-samples 4
python evaluate.py
```

**Test Qwen2.5-VL-7B on Standard (10 options) with 3 samples:**
```bash
python infer/infer_qwen.py qwen2.5-vl-7b cot "standard (10 options)" --max-samples 3
python evaluate.py
```

## Usage

### Command Format

**For Qwen models:**
```bash
python infer/infer_qwen.py <MODEL_NAME> <MODE> <SETTING> [--max-samples N]
```

**For GPT models:**
```bash
python infer/infer_gpt.py <MODEL_NAME> <MODE> <SETTING> [--max-samples N]
```

### Complete Examples

**Qwen Models:**
```bash
# Qwen2.5-VL-3B with Chain-of-Thought on Vision subset (validation cases only)
python infer/infer_qwen.py qwen2.5-vl-3b cot vision

# Qwen2.5-VL-3B on Standard (10 options) - validation cases only
python infer/infer_qwen.py qwen2.5-vl-3b cot "standard (10 options)"

# Qwen2.5-VL-7B with Direct mode on Standard 10 options (validation cases only)
python infer/infer_qwen.py qwen2.5-vl-7b direct "standard (10 options)"

# Qwen2.5-VL-32B with CoT on Standard 4 options, limit to 50 samples
python infer/infer_qwen.py qwen2.5-vl-32b cot "standard (4 options)" --max-samples 50
```

**Note**: By default, all evaluations run on validation cases only (577 samples for each subset).

**GPT Models:**
```bash
# GPT-4o-mini with Chain-of-Thought on Vision subset
python infer/infer_gpt.py gpt-4o-mini cot vision

# GPT-4o-mini with Direct mode on Standard 10 options
python infer/infer_gpt.py gpt4o-mini direct "standard (10 options)"

# Limit to 20 samples for testing
python infer/infer_gpt.py gpt-4o-mini cot vision --max-samples 20
```

### Running Evaluation

```bash
# Evaluate the most recent inference results
python evaluate.py
```

**Note**: The evaluate.py script automatically finds and evaluates the most recent inference results in the `./output/` directory.

## Parameters

### Model Names
- `qwen2.5-vl-3b`, `qwen2.5-vl-7b`, `qwen2.5-vl-32b`
- `gpt-4o-mini`, `gpt4o-mini`

### Modes
- `cot` - Chain-of-Thought reasoning (step-by-step thinking)
- `direct` - Direct answer without explicit reasoning steps

### Settings (Dataset Subsets)
- `vision` - Questions embedded in images (577 validation samples)
- `"standard (4 options)"` - Traditional 4-choice questions (577 validation samples)
- `"standard (10 options)"` - 10-choice questions (577 validation samples, more challenging)

## Output Files

### Inference Results
- **Location**: `./output/`
- **Format**: JSONL (one JSON object per line)
- **Filename**: `mmmu_pro_{model}_{mode}_{setting}_{timestamp}.jsonl`
- **Content**: Each line contains:
  ```json
  {
    "id": "sample_id",
    "subset": "vision",
    "subject": "Physics",
    "question": "...",
    "ground_truth": "A",
    "model_response": "...",
    "extracted_answer": "B",
    "inference_time": 2.34,
    "model_name": "qwen2.5-vl-3b",
    "mode": "cot",
    "setting": "vision"
  }
  ```

### Evaluation Results
- **Report**: `mmmu_pro_evaluation_report_{timestamp}.txt`
- **Metrics**: `mmmu_pro_evaluation_metrics_{timestamp}.json`
- **Details**: `mmmu_pro_evaluation_{timestamp}_detailed_results.csv`

## Configuration

Model configurations are defined in `../../config/config.toml`:

```toml
[llm.qwen2_5_vl_32b]
model = "qwen2.5-vl-32b-instruct"
api_type = "dashscope"
temperature = 0.01
# ... other MMMU PRO settings

[llm.gpt4o_mini]
model = "gpt-4o-mini"
api_type = "openai"
temperature = 0.0
# ... other settings
```

## Common Test Commands

### Quick Testing (Small Samples)

**Test with a small sample first:**
```bash
# Test with 3 samples to verify setup
python infer/infer_qwen.py qwen2.5-vl-3b cot vision --max-samples 3
```

### Production Testing (Full Evaluation on Validation Cases)

```bash
# Full evaluation on validation cases (577 samples each)
python infer/infer_qwen.py qwen2.5-vl-3b cot vision
python infer/infer_qwen.py qwen2.5-vl-3b cot "standard (10 options)"

# Evaluate results
python evaluate.py
```

## Example Workflow

```bash
# 1. Navigate to directory
cd src/multi-agent/benchmark_evaluation/mmmu_pro

# 2. Set environment variables
export DASHSCOPE_API_KEY="your_key"
export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

# 3. Test with small sample first
python infer/infer_qwen.py qwen2.5-vl-3b cot vision --max-samples 3

# 4. Evaluate test results
python evaluate.py

# 5. Run full validation evaluation (if testing was successful)
python infer/infer_qwen.py qwen2.5-vl-3b cot vision
python infer/infer_qwen.py qwen2.5-vl-3b cot "standard (10 options)"

# 6. Final evaluation
python evaluate.py
```

## Performance Expectations

Based on the official MMMU PRO leaderboard and our testing:
- **GPT-4o-mini**: ~37.6% overall accuracy
- **Qwen2.5-VL-3B**: ~25-30% accuracy (fast and reliable)
- **Qwen2.5-VL-7B**: ~23-38% accuracy (varies with parameters)
- **Qwen2.5-VL-32B**: Higher accuracy but slower inference

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```
   ValueError: DASHSCOPE_API_KEY environment variable not set
   ```
   Solution: Set the appropriate environment variable

2. **Dataset Download Issues**
   - Ensure internet connection
   - Check Hugging Face datasets access
   - Try clearing cache: `rm -rf ./cache`

3. **Model Loading Errors**
   - Verify model name spelling
   - Check API endpoints and keys
   - Review config.toml settings

### Debug Mode

Add `--max-samples 1` to test with a single sample:
```bash
python infer/infer_qwen.py qwen2.5-vl-3b direct vision --max-samples 1
```

## Future Extensions

This framework is designed to support:
- Additional model types
- Multi-agent flow integration
- Custom evaluation metrics
- Batch processing optimizations

## References

- [MMMU PRO Paper](https://arxiv.org/abs/2409.02813)
- [MMMU PRO Dataset](https://huggingface.co/datasets/MMMU/MMMU_Pro)
- [Official GitHub Repository](https://github.com/MMMU-Benchmark/MMMU/tree/main/mmmu-pro)
