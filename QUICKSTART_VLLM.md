# Quick Start: vLLM with Batch-Invariant Ops

Get deterministic outputs from vLLM at `temperature=0` using batch-invariant operations.

## 🚀 Fastest Start (Docker Compose)

```bash
# Build and start vLLM server
make vllm-compose-up

# Or run in background
make vllm-compose-up-detached

# View logs
make vllm-compose-logs

# Stop the server
make vllm-compose-down
```

The server will be available at `http://localhost:8000` after the model downloads.

## 📦 Alternative: Docker CLI

```bash
# Build the image
make vllm-build

# Run the server
make vllm-run

# Or run in background
make vllm-run-detached

# Stop the server
make vllm-stop
```

## 🧪 Test Determinism

After the server is running, verify that outputs are deterministic:

```bash
# Requires httpx: pip install httpx
make vllm-test
```

Expected output:

```
Testing determinism with 100 requests (10 concurrent)
...
✓ SUCCESS: All outputs are identical (deterministic)!
```

## 💬 Make a Request

### Using curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role":"user","content":"What is deterministic inference?"}],
    "temperature": 0,
    "max_tokens": 100
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
    },
)

print(response.json()["choices"][0]["message"]["content"])
```

## ⚙️ Configuration

### Use a Different Model

```bash
# With Makefile
make vllm-run VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2

# Or edit docker-compose.yml and change MODEL_NAME
```

### Change Port

```bash
make vllm-run VLLM_PORT=8080
```

### For Gated Models (e.g., Llama)

1. Get your HuggingFace token from https://huggingface.co/settings/tokens
2. Edit `vllm/docker-compose.yml` and add:
   ```yaml
   environment:
     - HF_TOKEN=your_token_here
   ```

## 📊 Available Commands

Run `make help` to see all available commands:

```
vllm-build                Build vLLM Docker image
vllm-run                  Run vLLM server with batch-invariant ops
vllm-run-detached         Run vLLM server in detached mode
vllm-compose-up           Start vLLM with docker-compose
vllm-compose-up-detached  Start vLLM with docker-compose (detached)
vllm-compose-down         Stop vLLM docker-compose services
vllm-compose-logs         View vLLM docker-compose logs
vllm-test                 Test vLLM determinism (requires running server)
vllm-stop                 Stop running vLLM container
vllm-logs                 Show logs from vLLM container
vllm-shell                Open shell in vLLM container
vllm-info                 Show vLLM configuration
```

## 🔍 How It Works

This setup ensures deterministic inference by:

1. **FlexAttention Backend**: Forces vLLM to use PyTorch's FlexAttention, which is deterministic by design
2. **Batch-Invariant Kernels**: Replaces PyTorch operations (`mm`, `addmm`, `log_softmax`, `mean`) with Triton kernels that have fixed reduction order
3. **Auto-initialization**: `sitecustomize.py` automatically enables batch-invariant mode in all vLLM worker processes

This eliminates floating-point differences caused by dynamic batching that lead to different argmax selections during greedy decoding.

## 🐛 Troubleshooting

### Server won't start

**Check GPU availability:**

```bash
nvidia-smi
```

**Check Docker GPU support:**

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Non-deterministic outputs

Ensure:

- ✅ `temperature` is exactly `0` (not `0.0001` or similar)
- ✅ Environment variable `VLLM_ATTENTION_BACKEND=FLEX_ATTENTION` is set
- ✅ `sitecustomize.py` is loaded (check logs for confirmation)
- ✅ Using PyTorch 2.5+ (built into the Docker image)

### Model won't download

For gated models (Llama, etc.):

1. Accept the model terms on HuggingFace
2. Add your `HF_TOKEN` to the environment (see Configuration above)

### Out of memory

Reduce model size or context length:

```bash
make vllm-run VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct  # 8B instead of 70B
```

Or edit `docker-compose.yml` and set:

```yaml
environment:
  - MAX_MODEL_LEN=4096 # Reduce from 8192
```

## 📚 More Information

- Full documentation: [vllm/README.md](vllm/README.md)
- Batch-invariant ops: [README.md](README.md)
- Test scripts: [vllm/test_determinism.py](vllm/test_determinism.py)

## 📝 Example: Concurrent Inference

```python
import asyncio
import httpx

async def generate(prompt: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
            timeout=120,
        )
        return response.json()["choices"][0]["message"]["content"]

# Send 10 concurrent requests with the same prompt
results = await asyncio.gather(*[
    generate("What is 2+2?")
    for _ in range(10)
])

# All results should be identical
assert len(set(results)) == 1, "Outputs are not deterministic!"
print("✓ All 10 outputs are identical")
```

---

**Need help?** Check the full documentation in [vllm/README.md](vllm/README.md) or open an issue.
