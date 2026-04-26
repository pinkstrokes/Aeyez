# LLaVA-1.5-7B Benchmark Evaluation# LLaVA-1.5-7B Benchmark Evaluation# LLaVA-1.5-7B MMMU Evaluation



Comprehensive evaluation of LLaVA-1.5-7B model on multiple multimodal benchmarks using HuggingFace Transformers. This directory provides scripts for evaluating LLaVA on MMMU, MMMU Pro, and MIA-Bench.



## OverviewComprehensive evaluation of LLaVA-1.5-7B model on multiple multimodal benchmarks using HuggingFace Transformers. This directory provides scripts for evaluating LLaVA on MMMU, MMMU Pro, and MIA-Bench.Simple and direct evaluation of LLaVA-1.5-7B on MMMU benchmark using HuggingFace Transformers.



This module provides LLaVA-1.5-7B evaluation capabilities for:

- **MMMU (Massive Multi-discipline Multimodal Understanding)**: Traditional 4-choice questions across multiple subjects

- **MMMU Pro**: Enhanced benchmark with standard (4/10 options) and vision-embedded questions## Overview## Model Information

- **MIA-Bench**: Multimodal instruction-following evaluation benchmark



## Model Information

This module provides LLaVA-1.5-7B evaluation capabilities for:- **Model**: llava-hf/llava-1.5-7b-hf

- **Model**: llava-hf/llava-1.5-7b-hf

- **Location**: Auto-detected from HuggingFace cache or specify with `--model-path`- **MMMU (Massive Multi-discipline Multimodal Understanding)**: Traditional 4-choice questions across multiple subjects- **Location**: Auto-detected from HuggingFace cache or specify with `--model-path`

- **Framework**: HuggingFace Transformers (no vLLM needed)

- **Can run on**: CPU or GPU (GPU recommended for speed)- **MMMU Pro**: Enhanced benchmark with standard (4/10 options) and vision-embedded questions- **Framework**: HuggingFace Transformers (no vLLM needed)



## Setup- **MIA-Bench**: Multimodal instruction-following evaluation benchmark- **Can run on**: CPU or GPU (GPU recommended for speed)



### 1. Environment Setup



**Python Environment**: Recommended Python 3.11+## Model Information## Quick Start



```bash

# Activate your Python environment

conda activate your_env- **Model**: llava-hf/llava-1.5-7b-hf### 1. Test with 3 samples



# Install dependencies- **Location**: Auto-detected from HuggingFace cache or specify with `--model-path`

pip install -r requirements.txt

```- **Framework**: HuggingFace Transformers (no vLLM needed)```bash



### 2. Model Download- **Can run on**: CPU or GPU (GPU recommended for speed)python run_mmmu.py infer --max-samples 3



The model will be auto-detected from your HuggingFace cache. If not present, download it:```



```bash## Setup

# Download from HuggingFace (automatic on first run)

# Or pre-download manually:### 2. Run full validation

huggingface-cli download llava-hf/llava-1.5-7b-hf

```### 1. Environment Setup



### 3. Dataset Preparation```bash



Each benchmark has different data requirements:**Python Environment**: Recommended Python 3.11+# Step 1: Inference



**MMMU**: Place `MMMU_DEV_VAL.tsv` in `inputs/` directorypython run_mmmu.py infer --dataset MMMU_DEV_VAL



**MMMU Pro**: Auto-downloads from HuggingFace (`MMMU/MMMU_Pro`)```bash



**MIA-Bench**: Place dataset in `inputs/mia_bench/` or specify with `--data-path`# Activate your Python environment# Step 2: Evaluation



## Usageconda activate your_envpython run_mmmu.py eval \



### 1. MMMU Evaluation    --input-file outputs/inference_results.jsonl \



MMMU tests multimodal understanding across multiple academic disciplines.# Install dependencies    --output-file outputs/eval_results.csv



#### Inferencepip install -r requirements.txt```



```bash```

# Run full validation set

python run_mmmu.py infer --dataset MMMU_DEV_VAL### 3. Use specific sample range



# Custom output file### 2. Model Download

python run_mmmu.py infer --output-file outputs/mmmu_results.jsonl

``````bash



#### EvaluationThe model will be auto-detected from your HuggingFace cache. If not present, download it:python run_mmmu.py infer --start-sample 0 --end-sample 100



```bash```

# Evaluate inference results

python run_mmmu.py eval \```bash

    --input-file outputs/inference_results.jsonl \

    --output-file outputs/eval_results.csv# Download from HuggingFace (automatic on first run)## Arguments



# With custom evaluation model# Or pre-download manually:

python run_mmmu.py eval \

    --input-file outputs/inference_results.jsonl \huggingface-cli download llava-hf/llava-1.5-7b-hf### Inference (`infer` mode)

    --eval-model gpt-4o-mini \

    --api-type openai```

```

```bash

#### Key Arguments

### 3. Dataset Preparationpython run_mmmu.py infer [options]

**Inference:**

- `--model-path`: Path to LLaVA model (default: auto-detected)```

- `--dataset`: Dataset name (default: MMMU_DEV_VAL)

- `--output-file`: Output JSONL fileEach benchmark has different data requirements:

- `--max-tokens`: Max tokens to generate (default: 512)

- `--cpu`: Force CPU usage**Options:**



**Evaluation:****MMMU**: Place `MMMU_DEV_VAL.tsv` in `inputs/` directory- `--model-path`: Path to model (default: auto-detected from hf_cache)

- `--input-file`: Input inference results

- `--output-file`: Output CSV file- `--dataset`: Dataset name (default: MMMU_DEV_VAL)

- `--eval-model`: Evaluation model (default: qwen-flash)

- `--api-type`: API type (dash/openai)**MMMU Pro**: Auto-downloads from HuggingFace (`MMMU/MMMU_Pro`)- `--data-dir`: Data directory (default: ./inputs)



### 2. MMMU Pro Evaluation- `--output-file`: Output file (default: outputs/inference_results.jsonl)



MMMU Pro provides enhanced evaluation with multiple question formats.**MIA-Bench**: Place dataset in `inputs/mia_bench/` or specify with `--data-path`- `--max-samples`: Limit number of samples



#### Available Subsets- `--start-sample`: Start from sample N



- `standard (4 options)`: Traditional 4-choice questions (577 validation samples)## Quick Start- `--end-sample`: End at sample N

- `standard (10 options)`: Increased difficulty with 10 choices (577 validation samples)

- `vision`: Questions embedded in images (577 validation samples)- `--max-tokens`: Max tokens to generate (default: 512)



#### Inference### Test with Small Samples (Recommended First Step)- `--temperature`: Sampling temperature (default: 0.01)



```bash- `--cpu`: Force CPU usage (default: auto-detect GPU)

# Run standard (4 options) validation set

python run_mmmu_pro.py --subset "standard (4 options)" --validation-only```bash



# Run standard (10 options) validation set# Test MMMU with 3 samples### Evaluation (`eval` mode)

python run_mmmu_pro.py --subset "standard (10 options)" --validation-only

python run_mmmu.py infer --max-samples 3

# Run vision subset validation set (use specialized script)

python run_mmmu_pro_vision.py --validation-only```bash



# Custom output directory# Test MMMU Pro standard (4 options) with 3 samplespython run_mmmu.py eval [options]

python run_mmmu_pro.py --subset "standard (4 options)" --output-dir ./custom_output

```python run_mmmu_pro.py --subset "standard (4 options)" --max-samples 3```



#### Key Arguments



- `--subset`: Subset name (required): "standard (4 options)", "standard (10 options)", or "vision"# Test MIA-Bench with 5 samples**Options:**

- `--validation-only`: Process only validation samples (default: True)

- `--output-dir`: Output directory (default: ./output)python run_mia.py --max-samples 5- `--input-file`: Input inference results (default: outputs/inference_results.jsonl)

- `--model-path`: Custom model path

- `--max-new-tokens`: Max tokens (default: 512)```- `--output-file`: Output CSV file (default: outputs/eval_results.csv)



#### Evaluation- `--dataset`: Dataset name (default: MMMU_DEV_VAL)



MMMU Pro uses a separate evaluation script:## Benchmark-Specific Usage- `--eval-model`: Evaluation model (default: qwen-flash)



```bash- `--api-type`: API type for eval (default: dash)

# Navigate to parent mmmu_pro directory

cd ../mmmu_pro### 1. MMMU Evaluation- `--nproc`: Number of parallel processes (default: 4)



# Evaluate results

python evaluate.py

MMMU tests multimodal understanding across multiple academic disciplines.## Directory Structure

# Or with custom paths

python evaluate.py --input-dir ../llava/output --output-dir ./eval_results

```

#### Inference```

### 3. MIA-Bench Evaluation

llava/

MIA-Bench tests models' ability to strictly adhere to complex, compositional instructions.

```bash├── README.md                    # This file

#### Inference

# Quick test with 3 samples├── run_mmmu.py                  # Main inference & evaluation script

```bash

# Run full MIA-Bench datasetpython run_mmmu.py infer --max-samples 3├── common_utils.py              # Utilities

python run_mia.py --data-path inputs/mia_bench/

├── dataset_utils.py             # Dataset loading

# Custom configuration

python run_mia.py \# Process specific sample range├── eval_utils.py                # Evaluation functions

    --data-path inputs/mia_bench/ \

    --max-tokens 1024 \python run_mmmu.py infer --start-sample 0 --end-sample 100├── requirements.txt             # Dependencies

    --output-file outputs/mia_results.jsonl

```├── inputs/



#### Evaluation# Full validation set│   └── MMMU_DEV_VAL.tsv        # Dataset file



MIA-Bench evaluation requires GPT-4o for official scoring:python run_mmmu.py infer --dataset MMMU_DEV_VAL├── outputs/                     # Results



```bash│   ├── inference_results.jsonl

# Set OpenAI API key

export OPENAI_API_KEY="your_openai_api_key"# Custom output file│   ├── eval_results.csv



# Run evaluationpython run_mmmu.py infer --output-file outputs/mmmu_results.jsonl│   └── eval_results_acc.json

python eval_mia.py \

    --input-file outputs/mia_inference_results.jsonl \```└── image_cache/                 # Cached images

    --output-file outputs/mia_eval_results.json

```

# With custom GPT model

python eval_mia.py \#### Evaluation

    --input-file outputs/mia_inference_results.jsonl \

    --eval-model gpt-4o \## Output Format

    --nproc 8

``````bash



#### Key Arguments# Evaluate inference results### Inference Output (JSONL)



**Inference:**python run_mmmu.py eval \```json

- `--data-path`: Path to MIA-Bench dataset

- `--output-file`: Output JSONL file    --input-file outputs/inference_results.jsonl \{

- `--max-tokens`: Max tokens (default: 1024)

- `--model-path`: Custom model path    --output-file outputs/eval_results.csv  "question_id": 1,



**Evaluation:**  "annotation": {...},

- `--input-file`: Input inference results

- `--output-file`: Output JSON file# With custom evaluation model  "task": "MMMU_DEV_VAL",

- `--eval-model`: GPT model for evaluation (default: gpt-4o)

- `--nproc`: Parallel processes (default: 4)python run_mmmu.py eval \  "result": {"gen": "A"}



## Complete Workflow Examples    --input-file outputs/inference_results.jsonl \}



### MMMU Complete Pipeline    --eval-model gpt-4o-mini \```



```bash    --api-type openai

# 1. Run inference on full validation set

python run_mmmu.py infer --dataset MMMU_DEV_VAL```### Evaluation Output (CSV)



# 2. Run evaluation```csv

python run_mmmu.py eval

#### Argumentsindex,prediction,GT,hit,subject,split

# 3. View results

cat outputs/eval_results_acc.json1,A,A,1,Math,validation

cat outputs/eval_results.csv

```**Inference Options:**2,B,C,0,Physics,validation



### MMMU Pro Complete Pipeline- `--model-path`: Path to LLaVA model (default: auto-detected)```



```bash- `--dataset`: Dataset name (default: MMMU_DEV_VAL)

# 1. Run inference on all three subsets (validation sets)

python run_mmmu_pro.py --subset "standard (4 options)" --validation-only- `--data-dir`: Data directory (default: ./inputs)### Accuracy Metrics (JSON)

python run_mmmu_pro.py --subset "standard (10 options)" --validation-only

python run_mmmu_pro_vision.py --validation-only- `--output-file`: Output JSONL file (default: outputs/inference_results.jsonl)```json



# 2. Evaluate all results- `--max-samples`: Limit number of samples{

cd ../mmmu_pro

python evaluate.py- `--start-sample`: Start from sample N  "overall_accuracy": 0.42,



# 3. View results- `--end-sample`: End at sample N  "accuracy_by_split": {

cat output/evaluation_summary.json

```- `--max-tokens`: Max tokens to generate (default: 512)    "validation": 0.42,



### MIA-Bench Complete Pipeline- `--temperature`: Sampling temperature (default: 0.01)    "dev": 0.38



```bash- `--cpu`: Force CPU usage  }

# 1. Set up evaluation API

export OPENAI_API_KEY="your_openai_api_key"}



# 2. Run inference on full dataset**Evaluation Options:**```

python run_mia.py --data-path inputs/mia_bench/

- `--input-file`: Input inference results

# 3. Run evaluation

python eval_mia.py --input-file outputs/mia_inference_results.jsonl- `--output-file`: Output CSV file## Requirements



# 4. View results- `--eval-model`: Evaluation model (default: qwen-flash)

cat outputs/mia_eval_results.json

```- `--api-type`: API type (dash/openai)```bash



## Directory Structure- `--nproc`: Parallel processes (default: 4)pip install torch transformers pillow pandas numpy tqdm



``````

llava/

├── README.md                       # This file### 2. MMMU Pro Evaluation

├── run_mmmu.py                     # MMMU inference & evaluation

├── run_mmmu_pro.py                 # MMMU Pro standard subsets## Notes

├── run_mmmu_pro_vision.py          # MMMU Pro vision subset

├── run_mia.py                      # MIA-Bench inferenceMMMU Pro provides enhanced evaluation with multiple question formats.

├── eval_mmmu.py                    # MMMU evaluation utilities

├── eval_mia.py                     # MIA-Bench evaluation- **Device**: Auto-detects GPU, falls back to CPU

├── eval_utils.py                   # Common evaluation functions

├── dataset_utils.py                # Dataset loading utilities#### Available Subsets- **Memory**: ~14GB GPU RAM or ~28GB CPU RAM (FP32)

├── common_utils.py                 # Common utilities

├── requirements.txt                # Python dependencies- **Speed**: GPU is 10-20x faster than CPU

├── inputs/

│   ├── MMMU_DEV_VAL.tsv           # MMMU dataset- `standard (4 options)`: Traditional 4-choice questions (577 validation samples)- **Images**: Cached in `image_cache/` automatically

│   └── mia_bench/                 # MIA-Bench dataset

├── outputs/                        # Inference results- `standard (10 options)`: Increased difficulty with 10 choices (577 validation samples)- **Model**: Uses local model from hf_cache (no download)

│   ├── inference_results.jsonl    # MMMU results

│   ├── eval_results.csv           # MMMU evaluation- `vision`: Questions embedded in images (577 validation samples)

│   ├── eval_results_acc.json      # MMMU accuracy

│   ├── mia_inference_results.jsonl # MIA results## Troubleshooting

│   └── mia_eval_results.json      # MIA evaluation

├── output/                         # MMMU Pro results#### Inference

│   ├── mmmu_pro_standard_4_*.jsonl

│   ├── mmmu_pro_standard_10_*.jsonl### Out of Memory

│   └── mmmu_pro_vision_*.jsonl

└── image_cache/                    # Cached images```bashProcess in smaller batches:

```

# Quick test on standard (4 options)```bash

## Output Format

python run_mmmu_pro.py --subset "standard (4 options)" --max-samples 3python run_mmmu.py infer --max-samples 100

### MMMU Output (JSONL)

```

```json

{# Quick test on standard (10 options)

  "question_id": 1,

  "annotation": {python run_mmmu_pro.py --subset "standard (10 options)" --max-samples 3### Image Loading Errors

    "question": "What is the capital of France?",

    "options": ["A. London", "B. Paris", "C. Berlin", "D. Madrid"],Set environment variable:

    "answer": "B"

  },# Full validation on vision subset```bash

  "task": "MMMU_DEV_VAL",

  "result": {"gen": "B"}python run_mmmu_pro.py --subset vision --validation-onlyexport LMUData=/path/to/MPU-RL-clean/src/multi-agent/benchmark_evaluation/llava/inputs

}

``````



### MMMU Pro Output (JSONL)# Process specific sample range



```jsonpython run_mmmu_pro.py --subset vision --start-sample 0 --end-sample 100### Model Not Found

{

  "id": "validation_0001",Verify model exists in your HuggingFace cache:

  "subset": "standard (4 options)",

  "subject": "Math",# Custom output directory```bash

  "question": "Solve the equation: 2x + 3 = 7",

  "options": ["A. x=1", "B. x=2", "C. x=3", "D. x=4"],python run_mmmu_pro.py --subset "standard (4 options)" --output-dir ./custom_outputls $HF_HOME/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/

  "ground_truth": "B",

  "model_response": "Let me solve this step by step...",```# Or specify custom path with --model-path

  "extracted_answer": "B",

  "inference_time": 2.34```

}

```#### Vision Subset (Special Handling)



### MIA-Bench Output (JSONL)For the vision subset where questions are embedded in images, use the specialized script:



```json```bash

{# Run vision validation set

  "question_id": "mia_001",python run_mmmu_pro_vision.py --validation-only

  "prompt": "Describe the image following these instructions...",```

  "response": "Based on the image, I can see...",

  "ground_truth": "Expected answer format",#### Arguments

  "images": ["path/to/image.jpg"]

}- `--subset`: Subset name (required): "standard (4 options)", "standard (10 options)", or "vision"

```- `--validation-only`: Process only validation samples (default: True)

- `--all-samples`: Process all samples including test set

## Performance Expectations- `--max-samples`: Limit number of samples

- `--start-sample`: Start from sample N

Based on LLaVA-1.5-7B capabilities:- `--end-sample`: End at sample N

- `--output-dir`: Output directory (default: ./output)

| Benchmark | Expected Accuracy | Inference Speed | Notes |- `--model-path`: Custom model path

|-----------|------------------|----------------|-------|- `--max-new-tokens`: Max tokens (default: 512)

| **MMMU** | 30-35% | ~2-3s/sample | Baseline performance on 4-option questions |- `--temperature`: Sampling temperature (default: 0.01)

| **MMMU Pro (4 opt)** | 28-33% | ~2-3s/sample | Similar to MMMU |

| **MMMU Pro (10 opt)** | 15-20% | ~2-3s/sample | More challenging with 10 options |#### Evaluation

| **MMMU Pro (vision)** | 25-30% | ~3-4s/sample | Requires image+text understanding |

| **MIA-Bench** | 40-50% | ~3-5s/sample | Instruction-following focused |MMMU Pro uses a separate evaluation script:



*Note: Actual performance varies based on hardware and exact dataset version*```bash

# Navigate to parent mmmu_pro directory

## Troubleshootingcd ../mmmu_pro



### Common Issues# Evaluate results

python evaluate.py

#### 1. Model Not Found

# Or with custom paths

**Error**: `Model not found in cache`python evaluate.py --input-dir ../llava/output --output-dir ./eval_results

```

**Solution**: Verify model exists in your HuggingFace cache:

```bash### 3. MIA-Bench Evaluation

ls $HF_HOME/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/

# Or specify custom path with --model-pathMIA-Bench tests models' ability to strictly adhere to complex, compositional instructions.

```

#### Inference

#### 2. Image Loading Errors

```bash

**Error**: `Cannot load image` or `PIL.UnidentifiedImageError`# Run full MIA-Bench dataset

python run_mia.py --data-path inputs/mia_bench/

**Solution**: Set environment variable:

```bash# Custom configuration

export LMUData=/path/to/MPU-RL-clean/src/multi-agent/benchmark_evaluation/llava/inputspython run_mia.py \

```    --data-path inputs/mia_bench/ \

    --max-tokens 1024 \

#### 3. CUDA Out of Memory    --output-file outputs/mia_results.jsonl

```

**Error**: `RuntimeError: CUDA out of memory`

#### Evaluation

**Solutions**:

```bashMIA-Bench evaluation requires GPT-4o for official scoring:

# Use CPU instead (slower but no GPU memory issues)

python run_mmmu.py infer --cpu```bash

# Set OpenAI API key

# Or use smaller max-tokensexport OPENAI_API_KEY="your_openai_api_key"

python run_mmmu.py infer --max-tokens 256

```# Run evaluation

python eval_mia.py \

#### 4. Dataset Not Found    --input-file outputs/mia_inference_results.jsonl \

    --output-file outputs/mia_eval_results.json

**MMMU**: Ensure `inputs/MMMU_DEV_VAL.tsv` exists

# With custom GPT model

**MMMU Pro**: Will auto-download from HuggingFace on first runpython eval_mia.py \

    --input-file outputs/mia_inference_results.jsonl \

**MIA-Bench**: Download from [ml-mia-bench](https://github.com/apple/ml-mia-bench) and place in `inputs/mia_bench/`    --eval-model gpt-4o \

    --nproc 8

#### 5. Evaluation API Errors```



**For MIA-Bench evaluation**:#### Arguments

```bash

# Verify API key is set**Inference:**

echo $OPENAI_API_KEY- `--data-path`: Path to MIA-Bench dataset

- `--max-samples`: Limit number of samples

# Test API connection- `--start-sample`: Start from sample N

python -c "import openai; print('API key valid')"- `--end-sample`: End at sample N

```- `--output-file`: Output JSONL file

- `--max-tokens`: Max tokens (default: 1024)

### Debug Mode- `--temperature`: Sampling temperature (default: 0.01)

- `--model-path`: Custom model path

```bash

# Check detailed logs**Evaluation:**

tail -f outputs/*.log- `--input-file`: Input inference results

- `--output-file`: Output JSON file

# Verify dataset loading- `--eval-model`: GPT model for evaluation (default: gpt-4o)

python -c "from dataset_utils import load_mmmu_dataset; print('Dataset loaded:', len(load_mmmu_dataset('inputs/MMMU_DEV_VAL.tsv')), 'samples')"- `--nproc`: Parallel processes (default: 4)

```

## Complete Workflow Examples

## Comparison with Other Approaches

### MMMU Complete Pipeline

### LLaVA vs Multi-Agent Flow

```bash

| Feature | LLaVA Direct (This Module) | Multi-Agent Flow |# 1. Run inference on full validation set

|---------|---------------------------|------------------|python run_mmmu.py infer --dataset MMMU_DEV_VAL

| **Architecture** | Single model inference | TranslatorAgent + ReasoningAgent |

| **Speed** | Faster (~2-3s/sample) | Slower (~5-10s/sample) |# 2. Run evaluation

| **Accuracy** | Baseline | Potentially higher with refinement |python run_mmmu.py eval

| **Setup** | Simple - just model | Requires vLLM cluster |

| **Use Case** | Quick evaluation, baseline | Research, iterative refinement |# 3. View results

cat outputs/eval_results_acc.json

### LLaVA vs API-based Models (Qwen, GPT)cat outputs/eval_results.csv

```

| Feature | LLaVA-1.5-7B | Qwen2.5-VL-7B | GPT-4o-mini |

|---------|--------------|---------------|-------------|### MMMU Pro Complete Pipeline

| **Deployment** | Local | API (DashScope) | API (OpenAI) |

| **Cost** | Free (GPU needed) | Pay per token | Pay per token |```bash

| **Customization** | Full control | Limited | Limited |# 1. Run inference on all three subsets (validation sets)

| **Performance** | Good baseline | Better | Best |python run_mmmu_pro.py --subset "standard (4 options)" --validation-only

python run_mmmu_pro.py --subset "standard (10 options)" --validation-only

## Best Practicespython run_mmmu_pro_vision.py --validation-only



1. **Monitor GPU memory** usage with `nvidia-smi` during inference# 2. Evaluate all results

2. **Use appropriate temperature**: Lower (0.01) for accuracy taskscd ../mmmu_pro

3. **Save intermediate results** regularly when processing large datasetspython evaluate.py

4. **Verify dataset format** before running full evaluation

5. **Use appropriate evaluation metrics** for each benchmark# 3. View results

cat output/evaluation_summary.json

## Citation```



If you use this evaluation code, please cite the respective benchmarks:### MIA-Bench Complete Pipeline



**MMMU**:```bash

```bibtex# 1. Set up evaluation API

@article{yue2023mmmu,export OPENAI_API_KEY="your_openai_api_key"

  title={MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark},

  author={Yue, Xiang and others},# 2. Run inference on full dataset

  journal={arXiv preprint arXiv:2311.16502},python run_mia.py --data-path inputs/mia_bench/

  year={2023}

}# 3. Run evaluation

```python eval_mia.py --input-file outputs/mia_inference_results.jsonl



**MMMU Pro**:# 4. View results

```bibtexcat outputs/mia_eval_results.json

@article{yue2024mmmupro,```

  title={MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark},

  author={Yue, Xiang and others},## Directory Structure

  year={2024}

}```

```llava/

├── README.md                       # This file

**MIA-Bench**:├── run_mmmu.py                     # MMMU inference & evaluation

```bibtex├── run_mmmu_pro.py                 # MMMU Pro standard subsets

@article{tu2024mia,├── run_mmmu_pro_vision.py          # MMMU Pro vision subset

  title={MIA-Bench: Towards Better Instruction Following Evaluation for Multimodal LLMs},├── run_mia.py                      # MIA-Bench inference

  author={Tu, Yusu and others},├── eval_mmmu.py                    # MMMU evaluation utilities

  year={2024}├── eval_mia.py                     # MIA-Bench evaluation

}├── eval_utils.py                   # Common evaluation functions

```├── dataset_utils.py                # Dataset loading utilities

├── common_utils.py                 # Common utilities

**LLaVA**:├── requirements.txt                # Python dependencies

```bibtex├── inputs/

@misc{liu2023llava,│   ├── MMMU_DEV_VAL.tsv           # MMMU dataset

  title={Visual Instruction Tuning},│   └── mia_bench/                 # MIA-Bench dataset

  author={Liu, Haotian and others},├── outputs/                        # Inference results

  year={2023}│   ├── inference_results.jsonl    # MMMU results

}│   ├── eval_results.csv           # MMMU evaluation

```│   ├── eval_results_acc.json      # MMMU accuracy

│   ├── mia_inference_results.jsonl # MIA results

## Additional Resources│   └── mia_eval_results.json      # MIA evaluation

├── output/                         # MMMU Pro results

- **MMMU Dataset**: [HuggingFace](https://huggingface.co/datasets/MMMU/MMMU)│   ├── mmmu_pro_standard_4_*.jsonl

- **MMMU Pro Dataset**: [HuggingFace](https://huggingface.co/datasets/MMMU/MMMU_Pro)│   ├── mmmu_pro_standard_10_*.jsonl

- **MIA-Bench**: [GitHub](https://github.com/apple/ml-mia-bench)│   └── mmmu_pro_vision_*.jsonl

- **LLaVA Model**: [HuggingFace](https://huggingface.co/llava-hf/llava-1.5-7b-hf)└── image_cache/                    # Cached images

- **LLaVA Project**: [GitHub](https://github.com/haotian-liu/LLaVA)

```

## Support

## Output Format

For issues specific to:

- **LLaVA model**: Check [LLaVA GitHub Issues](https://github.com/haotian-liu/LLaVA/issues)### MMMU Output (JSONL)

- **MMMU/MMMU Pro**: Check [MMMU GitHub](https://github.com/MMMU-Benchmark/MMMU)

- **MIA-Bench**: Check [MIA-Bench GitHub](https://github.com/apple/ml-mia-bench)```json

- **This integration**: Open an issue in your project repository{

  "question_id": 1,
  "annotation": {
    "question": "What is the capital of France?",
    "options": ["A. London", "B. Paris", "C. Berlin", "D. Madrid"],
    "answer": "B"
  },
  "task": "MMMU_DEV_VAL",
  "result": {"gen": "B"}
}
```

### MMMU Pro Output (JSONL)

```json
{
  "id": "validation_0001",
  "subset": "standard (4 options)",
  "subject": "Math",
  "question": "Solve the equation: 2x + 3 = 7",
  "options": ["A. x=1", "B. x=2", "C. x=3", "D. x=4"],
  "ground_truth": "B",
  "model_response": "Let me solve this step by step...",
  "extracted_answer": "B",
  "inference_time": 2.34
}
```

### MIA-Bench Output (JSONL)

```json
{
  "question_id": "mia_001",
  "prompt": "Describe the image following these instructions...",
  "response": "Based on the image, I can see...",
  "ground_truth": "Expected answer format",
  "images": ["path/to/image.jpg"]
}
```

## Performance Expectations

Based on LLaVA-1.5-7B capabilities:

| Benchmark | Expected Accuracy | Inference Speed | Notes |
|-----------|------------------|----------------|-------|
| **MMMU** | 30-35% | ~2-3s/sample | Baseline performance on 4-option questions |
| **MMMU Pro (4 opt)** | 28-33% | ~2-3s/sample | Similar to MMMU |
| **MMMU Pro (10 opt)** | 15-20% | ~2-3s/sample | More challenging with 10 options |
| **MMMU Pro (vision)** | 25-30% | ~3-4s/sample | Requires image+text understanding |
| **MIA-Bench** | 40-50% | ~3-5s/sample | Instruction-following focused |

*Note: Actual performance varies based on hardware and exact dataset version*

## Troubleshooting

### Common Issues

#### 1. Model Not Found

**Error**: `Model not found in cache`

**Solution**: Verify model exists in your HuggingFace cache:
```bash
ls $HF_HOME/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/
# Or specify custom path with --model-path
```

#### 2. Image Loading Errors

**Error**: `Cannot load image` or `PIL.UnidentifiedImageError`

**Solution**: Set environment variable:
```bash
export LMUData=/path/to/MPU-RL-clean/src/multi-agent/benchmark_evaluation/llava/inputs
```

#### 3. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Use CPU instead (slower but no GPU memory issues)
python run_mmmu.py infer --cpu

# Or use smaller max-tokens
python run_mmmu.py infer --max-tokens 256
```

#### 4. Dataset Not Found

**MMMU**: Ensure `inputs/MMMU_DEV_VAL.tsv` exists

**MMMU Pro**: Will auto-download from HuggingFace on first run

**MIA-Bench**: Download from [ml-mia-bench](https://github.com/apple/ml-mia-bench) and place in `inputs/mia_bench/`

#### 5. Evaluation API Errors

**For MIA-Bench evaluation**:
```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Test API connection
python -c "import openai; print('API key valid')"
```

### Debug Mode

```bash
# Check detailed logs
tail -f outputs/*.log

# Verify dataset loading
python -c "from dataset_utils import load_mmmu_dataset; print('Dataset loaded:', len(load_mmmu_dataset('inputs/MMMU_DEV_VAL.tsv')), 'samples')"
```

## Comparison with Other Approaches

### LLaVA vs Multi-Agent Flow

| Feature | LLaVA Direct (This Module) | Multi-Agent Flow |
|---------|---------------------------|------------------|
| **Architecture** | Single model inference | TranslatorAgent + ReasoningAgent |
| **Speed** | Faster (~2-3s/sample) | Slower (~5-10s/sample) |
| **Accuracy** | Baseline | Potentially higher with refinement |
| **Setup** | Simple - just model | Requires vLLM cluster |
| **Use Case** | Quick evaluation, baseline | Research, iterative refinement |

### LLaVA vs API-based Models (Qwen, GPT)

| Feature | LLaVA-1.5-7B | Qwen2.5-VL-7B | GPT-4o-mini |
|---------|--------------|---------------|-------------|
| **Deployment** | Local | API (DashScope) | API (OpenAI) |
| **Cost** | Free (GPU needed) | Pay per token | Pay per token |
| **Customization** | Full control | Limited | Limited |
| **Performance** | Good baseline | Better | Best |

## Best Practices

1. **Always test with small samples first** before running full evaluation
2. **Monitor GPU memory** usage with `nvidia-smi` during inference
3. **Use appropriate temperature**: Lower (0.01) for accuracy, higher (0.2-0.5) for creativity
4. **Save intermediate results** regularly when processing large datasets
5. **Verify dataset format** before running full evaluation
6. **Use appropriate evaluation metrics** for each benchmark

## Citation

If you use this evaluation code, please cite the respective benchmarks:

**MMMU**:
```bibtex
@article{yue2023mmmu,
  title={MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark},
  author={Yue, Xiang and others},
  journal={arXiv preprint arXiv:2311.16502},
  year={2023}
}
```

**MMMU Pro**:
```bibtex
@article{yue2024mmmupro,
  title={MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark},
  author={Yue, Xiang and others},
  year={2024}
}
```

**MIA-Bench**:
```bibtex
@article{tu2024mia,
  title={MIA-Bench: Towards Better Instruction Following Evaluation for Multimodal LLMs},
  author={Tu, Yusu and others},
  year={2024}
}
```

**LLaVA**:
```bibtex
@misc{liu2023llava,
  title={Visual Instruction Tuning},
  author={Liu, Haotian and others},
  year={2023}
}
```

## Additional Resources

- **MMMU Dataset**: [HuggingFace](https://huggingface.co/datasets/MMMU/MMMU)
- **MMMU Pro Dataset**: [HuggingFace](https://huggingface.co/datasets/MMMU/MMMU_Pro)
- **MIA-Bench**: [GitHub](https://github.com/apple/ml-mia-bench)
- **LLaVA Model**: [HuggingFace](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- **LLaVA Project**: [GitHub](https://github.com/haotian-liu/LLaVA)

## Support

For issues specific to:
- **LLaVA model**: Check [LLaVA GitHub Issues](https://github.com/haotian-liu/LLaVA/issues)
- **MMMU/MMMU Pro**: Check [MMMU GitHub](https://github.com/MMMU-Benchmark/MMMU)
- **MIA-Bench**: Check [MIA-Bench GitHub](https://github.com/apple/ml-mia-bench)
- **This integration**: Open an issue in your project repository
