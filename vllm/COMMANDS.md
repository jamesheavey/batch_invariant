# vLLM Quick Command Reference

## 🚀 Start Server

```bash
# Easiest (recommended)
make vllm-compose-up

# Or in background
make vllm-compose-up-detached

# Using plain Docker
make vllm-run
```

## 🧪 Test Determinism

```bash
# Run test suite
make vllm-test

# Or directly
python vllm/test_determinism.py --requests 100 --concurrency 10
```

## 📊 Monitor

```bash
# View logs (compose)
make vllm-compose-logs

# View logs (docker)
make vllm-logs

# Health check
curl http://localhost:8000/health
```

## 🛑 Stop Server

```bash
# Compose
make vllm-compose-down

# Docker
make vllm-stop
```

## 💬 Make Requests

### curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role":"user","content":"Hello!"}],
    "temperature": 0
  }'
```

### Python

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

### Example Script

```bash
python vllm/example_client.py
```

## ⚙️ Configuration

### Change Model

```bash
# Edit docker-compose.yml
MODEL_NAME: mistralai/Mistral-7B-Instruct-v0.2

# Or with make
make vllm-run VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```

### Change Port

```bash
make vllm-run VLLM_PORT=8080
```

### Use Gated Model

```yaml
# Add to docker-compose.yml
environment:
  - HF_TOKEN=your_token_here
```

## 🔧 Debug

```bash
# Open shell in container
make vllm-shell

# Check GPU
nvidia-smi

# Check if batch-invariant mode is loaded
# Look for sitecustomize message in logs
make vllm-logs | grep sitecustomize
```

## 📦 Build & Deploy

```bash
# Build image
make vllm-build

# Push to registry (configure USERNAME in Makefile)
make vllm-push

# Show config
make vllm-info
```

## 🆘 Common Issues

### Server won't start

```bash
# Check GPU
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Out of memory

```bash
# Use smaller model
make vllm-run VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct

# Or reduce context length in docker-compose.yml
MAX_MODEL_LEN: 4096
```

### Non-deterministic outputs

- Ensure `temperature: 0` (exactly 0)
- Check logs for sitecustomize.py loading
- Verify VLLM_ATTENTION_BACKEND=FLEX_ATTENTION

## 📚 More Help

- [QUICKSTART_VLLM.md](../QUICKSTART_VLLM.md) - Quick start guide
- [README.md](README.md) - Full documentation
- [example_client.py](example_client.py) - Python examples
- [test_determinism.py](test_determinism.py) - Test script

## 💡 Tips

1. Use `make vllm-compose-up-detached` for background running
2. Cache HuggingFace models with volume mount (done by default)
3. Test determinism regularly with `make vllm-test`
4. Monitor logs with `make vllm-compose-logs`
5. Use `temperature=0` for deterministic outputs
