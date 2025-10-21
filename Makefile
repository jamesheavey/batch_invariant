.PHONY: install serve serve-qwen3-8b serve-qwen3-30b test clean

VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
VLLM = $(VENV)/bin/vllm

# Install dependencies including vLLM from the specific PR
install:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "Installing PyTorch nightly (CUDA 13.0)..."
	$(PIP) install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
	@echo "Installing package dependencies..."
	$(PIP) install -e .

# Serve vLLM inference endpoint with Qwen3-8B (default)
serve: serve-qwen3-8b

# Serve Qwen3-8B model on localhost:8000
serve-qwen3-8b:
	@echo "Starting vLLM server with Qwen/Qwen3-8B on http://localhost:8000"
	@echo "Setting VLLM_BATCH_INVARIANT=1 for deterministic inference"
	VLLM_BATCH_INVARIANT=1 $(VLLM) serve Qwen/Qwen3-8B --enforce-eager

# Serve Qwen3-30B-A3B model on localhost:8000
serve-qwen3-30b:
	@echo "Starting vLLM server with Qwen/Qwen3-30B-A3B on http://localhost:8000"
	@echo "Setting VLLM_BATCH_INVARIANT=1 for deterministic inference"
	VLLM_BATCH_INVARIANT=1 $(VLLM) serve Qwen/Qwen3-30B-A3B --enforce-eager

# Serve with custom model (usage: make serve-custom MODEL=model_name)
serve-custom:
	@echo "Starting vLLM server with $(MODEL) on http://localhost:8000"
	@echo "Setting VLLM_BATCH_INVARIANT=1 for deterministic inference"
	VLLM_BATCH_INVARIANT=1 $(VLLM) serve $(MODEL) --enforce-eager

# Run the deterministic inference test
test:
	$(PYTHON) deterministic_vllm_inference.py

