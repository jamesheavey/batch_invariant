.PHONY: install serve serve-qwen3-8b serve-qwen3-30b test clean

# Install dependencies including vLLM from the specific PR
install:
	pip install -e .

# Serve vLLM inference endpoint with Qwen3-8B (default)
serve: serve-qwen3-8b

# Serve Qwen3-8B model on localhost:8000
serve-qwen3-8b:
	@echo "Starting vLLM server with Qwen/Qwen3-8B on http://localhost:8000"
	@echo "Setting VLLM_BATCH_INVARIANT=1 for deterministic inference"
	VLLM_BATCH_INVARIANT=1 vllm serve Qwen/Qwen3-8B --enforce-eager

# Serve Qwen3-30B-A3B model on localhost:8000
serve-qwen3-30b:
	@echo "Starting vLLM server with Qwen/Qwen3-30B-A3B on http://localhost:8000"
	@echo "Setting VLLM_BATCH_INVARIANT=1 for deterministic inference"
	VLLM_BATCH_INVARIANT=1 vllm serve Qwen/Qwen3-30B-A3B --enforce-eager

# Serve with custom model (usage: make serve-custom MODEL=model_name)
serve-custom:
	@echo "Starting vLLM server with $(MODEL) on http://localhost:8000"
	@echo "Setting VLLM_BATCH_INVARIANT=1 for deterministic inference"
	VLLM_BATCH_INVARIANT=1 vllm serve $(MODEL) --enforce-eager

# Run the deterministic inference test
test:
	python deterministic_vllm_inference.py

# Clean Python cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

