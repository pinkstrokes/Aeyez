## Quick Commands After Setup

```bash
python ./src/multi-agent/benchmark_evaluation/mmmu/run_mmmu_flow.py infer --validation-only --vision-port 8000 --text-port 8001 --start-sample 338 --end-sample 451

python ./src/multi-agent/benchmark_evaluation/mmmu/run_mmmu_flow.py infer --dev-only --vision-port 8002 --text-port 8003 --start-sample 122

python ./src/multi-agent/benchmark_evaluation/mmmu/run_mmmu_flow.py eval --input-file logs/mmmu/full_mmmu/inference_results.jsonl
``` 

## VLLM Setup
```bash
conda create -n vllm_infer python=3.12
conda activate vllm_infer
```

https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#create-a-new-python-environment
install wheels:
```bash
uv pip install vllm --torch-backend=auto
```

## Quick Start
### 1. Start VLLM Servers
**Single Instance (Default Ports 8000/8001)**:
```bash
./sbatch_vllm.sh
```

**Multiple Instances for Parallel Processing**:
```bash
# First instance (ports 8000/8001)
./sbatch_vllm.sh 8000 8001

# Second instance (ports 8002/8003)
./sbatch_vllm.sh 8002 8003

# Third instance (ports 8004/8005)
./sbatch_vllm.sh 8004 8005
```

Monitor jobs:
```bash
squeue -u $USER
```

Cancel all jobs if needed:
```bash
scancel -u $USER
```

### 2. Basic Flow Inference

**Single Instance (Default Ports)**:
```bash
cd MPU-RL/src/multi-agent/benchmark_evaluation/mmmu
python run_mmmu_flow.py infer --max-samples 5
```

**Specify Custom Ports**:
```bash
# Use specific port pair
python run_mmmu_flow.py infer --max-samples 5 --vision-port 8002 --text-port 8003
```

**Output**:
- `inference_results.jsonl` (raw results)
- `inference_results.csv` (automatically generated!)

### 3. Evaluate Flow Results
```bash
python run_mmmu_flow.py eval --input-file logs/mmmu/full_mmmu/inference_results.jsonl
```

## Port Management & Parallel Processing

### Port Range Convention
- **Instance 1**: Vision=8000, Text=8001 (default)
- **Instance 2**: Vision=8002, Text=8003
- **Instance 3**: Vision=8004, Text=8005
- **Instance 4**: Vision=8006, Text=8007