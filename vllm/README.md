# vLLM with Batch-Invariant Ops for Deterministic Inference

This directory contains Docker configuration and utilities for running vLLM with batch-invariant operations enabled, ensuring deterministic outputs at `temperature=0`.

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Build and start the server
docker-compose -f vllm/docker-compose.yml up --build

# The server will be available at http://localhost:8000
```

### Option 2: Docker CLI

```bash
# Build the image
docker build -f Dockerfile.vllm -t vllm-batch-invariant:latest .

# Run the container
docker run --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --shm-size 16g \
  -e MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
  vllm-batch-invariant:latest
```

### Option 3: Build from Source (Local Development)

```bash
# Install dependencies
pip install -e .

# Set required environment variables
export VLLM_ATTENTION_BACKEND=FLEX_ATTENTION
export CUBLAS_WORKSPACE_CONFIG=:16:8
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Copy sitecustomize.py to your working directory
cp vllm/sitecustomize.py .

# Start vLLM server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --dtype auto \
  --enforce-eager
```

## Testing Determinism

After starting the server, test that outputs are deterministic:

```bash
# Install test dependencies
pip install httpx

# Run the determinism test (sends 100 concurrent requests)
python vllm/test_determinism.py --requests 100 --concurrency 10

# Or test with the existing script
python vllm/deterministic_vllm_inference.py
```

Expected output:

```
Testing determinism with 100 requests (10 concurrent)
...
✓ SUCCESS: All outputs are identical (deterministic)!
```

## Making API Requests

### Using curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role":"user","content":"Explain batch invariance in one sentence."}],
    "temperature": 0,
    "top_p": 1,
    "n": 1
  }'
```

### Using Python

```python
import httpx

response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 0,
        "max_tokens": 100,
    },
    timeout=120,
)

print(response.json()["choices"][0]["message"]["content"])
```

## Configuration

### Environment Variables

You can customize the deployment by setting these environment variables:

| Variable                  | Default                            | Description                                  |
| ------------------------- | ---------------------------------- | -------------------------------------------- |
| `MODEL_NAME`              | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model to serve                   |
| `MAX_MODEL_LEN`           | `8192`                             | Maximum sequence length                      |
| `HOST`                    | `0.0.0.0`                          | Server host                                  |
| `PORT`                    | `8000`                             | Server port                                  |
| `VLLM_ATTENTION_BACKEND`  | `FLEX_ATTENTION`                   | Attention backend (required for determinism) |
| `CUBLAS_WORKSPACE_CONFIG` | `:16:8`                            | cuBLAS workspace config                      |
| `HF_TOKEN`                | (none)                             | HuggingFace token for gated models           |

### Using Different Models

Edit `docker-compose.yml` or set environment variable:

```bash
docker run --gpus all -p 8000:8000 \
  -e MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2 \
  vllm-batch-invariant:latest
```

## How It Works

1. **FlexAttention Backend**: vLLM is configured to use the FlexAttention backend, which supports deterministic execution
2. **Batch-Invariant Ops**: Custom Triton kernels replace PyTorch's non-deterministic operations (`mm`, `addmm`, `log_softmax`, `mean`) to ensure consistent reduction order regardless of batch shape
3. **Auto-initialization**: `sitecustomize.py` automatically enables batch-invariant mode in all vLLM worker processes on startup

This eliminates the subtle floating-point differences that cause different argmax selections during greedy decoding when dynamic batching is used.

## Troubleshooting

### Server won't start

Check GPU availability:

```bash
docker run --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Non-deterministic outputs

Ensure:

- Temperature is exactly `0`
- `VLLM_ATTENTION_BACKEND=FLEX_ATTENTION` is set
- `sitecustomize.py` is loaded (check container logs for confirmation)
- Using PyTorch 2.5+ (required for FlexAttention)

### Performance concerns

The batch-invariant kernels may have a small performance overhead (~5-15%) compared to standard PyTorch operations. This is a known tradeoff for deterministic execution and is expected to improve over time.

## Files

- `Dockerfile.vllm` - Production Docker image with vLLM and batch-invariant ops
- `docker-compose.yml` - Docker Compose configuration for easy deployment
- `sitecustomize.py` - Auto-enables batch-invariant mode in Python processes
- `test_determinism.py` - Concurrent test harness to verify deterministic outputs
- `deterministic_vllm_inference.py` - Original test script from Thinking Machines

## References

- [Thinking Machines Lab - Batch Invariant Ops](https://github.com/thinking-machines-lab/batch_invariant_ops)
- [vLLM Documentation](https://docs.vllm.ai/)
- [PyTorch FlexAttention](https://pytorch.org/blog/flexattention/)
