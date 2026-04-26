# MMMU PRO Multi-Agent Flow Evaluation

This module provides multi-agent flow evaluation for MMMU PRO benchmark using the IterativeRefinementFlow architecture with TranslatorAgent and TextOnlyReasoningAgent.

## Quick Start

```bash
# Example: Run vision subset with multi-agent flow
python src/multi-agent/benchmark_evaluation/mmmu_pro/run_mmmu_pro_flow.py infer "vision" --validation-only --vision-port 8002 --text-port 8003

# Example: Run standard (10 options) subset  
python src/multi-agent/benchmark_evaluation/mmmu_pro/run_mmmu_pro_flow.py infer "standard (10 options)" --validation-only --vision-port 8004 --text-port 8005
```

## Usage

### Basic Commands

```bash
# Navigate to directory
cd src/multi-agent/benchmark_evaluation/mmmu_pro

# Run inference on vision subset with flow
python run_mmmu_pro_flow.py infer "vision" --max-iterations 3

# Run inference on standard 10-option subset
python run_mmmu_pro_flow.py infer "standard (10 options)" --max-iterations 2

# Test with small sample first
python run_mmmu_pro_flow.py infer "vision" --max-samples 5 --verbose

# Evaluate results using existing evaluator
python run_mmmu_pro_flow.py eval

python run_mmmu_pro_flow.py infer "standard (10 options)" --vision-port 8000 --text-port 8001
```

### Advanced Options

```bash
# Custom sample range
python run_mmmu_pro_flow.py infer "vision" --start-sample 10 --end-sample 50

# Process all samples (not just validation)
python run_mmmu_pro_flow.py infer "vision" --all-samples

# Custom vLLM ports for cluster setup
python run_mmmu_pro_flow.py infer "vision" --vision-port 8000 --text-port 8001

# Skip vLLM setup (if already configured)
python run_mmmu_pro_flow.py infer "vision" --skip-vllm-setup

# Custom output directory
python run_mmmu_pro_flow.py infer "vision" --output-dir ./my_results
```

## Configuration

### Model Configuration
Models are configured in `../../config/config.toml`:

```toml
[llm.translator_api]
model = "Qwen/Qwen2.5-VL-7B-Instruct"
api_type = "vllm"
base_url = "http://localhost:8000/v1"

[llm.reasoning_api]
model = "Qwen/Qwen2.5-7B-Instruct"
api_type = "vllm"
base_url = "http://localhost:8001/v1"
```

### vLLM Cluster Setup
The script automatically sets up vLLM cluster access if needed:
- Vision model on port 8000 (for TranslatorAgent)
- Text model on port 8001 (for TextOnlyReasoningAgent)

## Output Files

### Flow Execution Logs
- **Location**: Automatic session directory in logs
- **Content**: Detailed execution traces, token usage, agent interactions
- **Format**: Structured JSON logs per question

### MMMU PRO Compatible Results
- **Location**: `./output/mmmu_pro_{subset}_{timestamp}.jsonl`
- **Format**: Compatible with existing `evaluate.py` script
- **Content**:
  ```json
  {
    "id": "validation_0001",
    "subset": "vision",
    "subject": "Physics",
    "question": "[Question embedded in image]",
    "ground_truth": "A",
    "model_response": "Looking at this image, I can see...",
    "extracted_answer": "B",
    "inference_time": 12.34,
    "model_name": "flow_system",
    "mode": "multi_agent_flow",
    "setting": "vision"
  }
  ```

## Comparison with Single Model Evaluation

| Feature | Single Model (`infer_qwen.py`) | Multi-Agent Flow (`run_mmmu_pro_flow.py`) |
|---------|---------------------------|------------------------------------------|
| **Architecture** | Direct model inference | TranslatorAgent + ReasoningAgent |
| **Reasoning** | Model's internal reasoning | Explicit multi-step refinement |
| **Error Recovery** | Basic retry | Advanced flow-level recovery |
| **Logging** | Basic result logging | Comprehensive session tracking |
| **Iterations** | Single pass | Configurable iterative refinement |
| **Token Tracking** | Per-request | Per-agent detailed tracking |

## Performance Expectations

Based on the multi-agent flow architecture:

- **Accuracy**: Expected improvement over single model due to iterative refinement
- **Latency**: ~2-3x slower than single model due to multi-step processing
- **Token Usage**: Higher due to multiple agent interactions
- **Robustness**: Better error handling and recovery

## Example Workflow

```bash
# 1. Quick test with small sample
python run_mmmu_pro_flow.py infer "vision" --max-samples 3 --verbose

# 2. Check logs and results
ls -la output/
cat session_info.txt  # Shows session directory

# 3. Run evaluation on test results
python run_mmmu_pro_flow.py eval

# 4. Full validation run (if test successful)
python run_mmmu_pro_flow.py infer "vision" --max-iterations 3

# 5. Run other subsets
python run_mmmu_pro_flow.py infer "standard (4 options)"
python run_mmmu_pro_flow.py infer "standard (10 options)"

# 6. Final comprehensive evaluation
python evaluate.py  # Uses existing MMMU PRO evaluator
```

## Architecture Details

### Flow Input Format
The script converts MMMU PRO samples to flow input:

```python
# Vision subset
"Please answer the question shown in the image.\n\nOptions:\nA. Option 1\nB. Option 2\nimage_path:/path/to/cached/image.jpg"

# Standard subsets
"What is the solution to this problem?\n\nOptions:\nA. Option 1\nB. Option 2\nC. Option 3\nD. Option 4"
```

### Agent Workflow
1. **Input Processing**: Sample → Flow Input Format
2. **Translation Step**: TranslatorAgent processes multimodal input
3. **Reasoning Step**: TextOnlyReasoningAgent performs logical reasoning
4. **Iterative Refinement**: Up to max_iterations rounds of improvement
5. **Output Generation**: Final answer with execution metadata

### Integration with Existing Infrastructure
- Uses `BenchmarkInfer` base class for consistency
- Leverages existing `FlowExecutor` and `LogSave` systems
- Compatible with existing `evaluate.py` for result analysis
- Follows same output format as single model evaluations

## Troubleshooting

### Common Issues

1. **vLLM Connection Issues**
   ```bash
   # Check cluster status
   python -c "from app.utils.vllm_setup import check_and_setup_vllm; check_and_setup_vllm()"
   ```

2. **Dataset Loading Errors**
   ```bash
   # Test dataset loading
   python -c "from dataset_utils import MMMUProDataset; MMMUProDataset().load_subsets(['vision'])"
   ```

3. **Memory Issues**
   - Reduce `--max-samples` for testing
   - Check vLLM GPU memory usage
   - Use smaller models in config

### Debug Mode
```bash
# Single sample with full logging
python run_mmmu_pro_flow.py infer "vision" --max-samples 1 --verbose

# Check session logs
tail -f logs/session_*/questions/*.json
```

## Integration Benefits

- **Consistency**: Same infrastructure as other benchmarks (MMMU, MIA)
- **Extensibility**: Easy to add new agent types or flow modifications
- **Monitoring**: Rich logging and token tracking
- **Robustness**: Advanced error handling and recovery mechanisms
- **Scalability**: Designed for cluster-based evaluation

This integration brings the power of multi-agent reasoning to MMMU PRO evaluation while maintaining compatibility with existing evaluation tools and infrastructure.