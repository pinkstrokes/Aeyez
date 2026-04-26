### Option 1: Separate Ports (Recommended)

Serve each model on a different port to avoid conflicts and enable independent scaling.

#### Serve Qwen2.5-VL-3B (Translator Agent)
```bash
# Terminal 1 - Vision-Language Model for Translator
# IMPORTANT: Use python -m vllm.entrypoints.openai.api_server for multi-modal models
# NOTE: --max-model-len is omitted to use model's native context length (32K+ tokens)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.45 \
    --disable-custom-all-reduce \
    --enforce-eager \
    --dtype float16 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

# Alternative using vllm serve (may not support multi-modal properly):
# vllm serve Qwen/Qwen2.5-VL-3B-Instruct --port 8000 ...
```

#### Serve Qwen3-8B (Text-Only Reasoning Agent)
```bash
# Terminal 2 - Text-Only Model for Reasoning
# NOTE: --max-model-len is omitted to use model's native context length (128K+ tokens)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --port 8001 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.45 \
    --disable-custom-all-reduce \
    --enforce-eager \
    --dtype float16 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

# Alternative using vllm serve (for text-only models):
# vllm serve Qwen/Qwen3-8B --port 8001 ...
```
