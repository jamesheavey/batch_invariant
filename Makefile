# Docker image configuration
IMAGE_NAME ?= llm-batch-invariant
REGISTRY ?= docker.io
USERNAME ?= jamesheavey
TAG ?= latest
DOCKERFILE ?= Dockerfile.llm
FULL_IMAGE = $(REGISTRY)/$(USERNAME)/$(IMAGE_NAME):$(TAG)

# Build platforms for multi-arch builds
PLATFORMS ?= linux/amd64,linux/arm64

.PHONY: help
help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

.PHONY: build
build: ## Build Docker image locally
	docker build -f $(DOCKERFILE) -t $(FULL_IMAGE) .

.PHONY: buildx-setup
buildx-setup: ## Setup buildx builder
	docker buildx create --name batch-invariant-builder --use --bootstrap || docker buildx use batch-invariant-builder

.PHONY: buildx
buildx: buildx-setup ## Build multi-platform image with buildx
	docker buildx build \
		--platform $(PLATFORMS) \
		-f $(DOCKERFILE) \
		-t $(FULL_IMAGE) \
		--load \
		.

.PHONY: buildx-push
buildx-push: buildx-setup ## Build and push multi-platform image
	docker buildx build \
		--platform $(PLATFORMS) \
		-f $(DOCKERFILE) \
		-t $(FULL_IMAGE) \
		--push \
		.

.PHONY: push
push: ## Push image to registry
	docker push $(FULL_IMAGE)

.PHONY: run
run: ## Run the container
	docker run --rm --gpus all $(FULL_IMAGE)

.PHONY: run-interactive
run-interactive: ## Run container interactively
	docker run --rm -it --gpus all $(FULL_IMAGE) /bin/bash

.PHONY: clean
clean: ## Remove local images
	docker rmi $(FULL_IMAGE) || true

.PHONY: info
info: ## Show build configuration
	@echo "Image:      $(FULL_IMAGE)"
	@echo "Dockerfile: $(DOCKERFILE)"
	@echo "Platforms:  $(PLATFORMS)"
	@echo "Registry:   $(REGISTRY)"
	@echo "Username:   $(USERNAME)"

# vLLM targets
VLLM_IMAGE_NAME ?= vllm-batch-invariant
VLLM_TAG ?= latest
VLLM_FULL_IMAGE = $(REGISTRY)/$(USERNAME)/$(VLLM_IMAGE_NAME):$(VLLM_TAG)
VLLM_MODEL ?= meta-llama/Llama-3.1-8B-Instruct
VLLM_PORT ?= 8000

.PHONY: vllm-build
vllm-build: ## Build vLLM Docker image
	docker build -f Dockerfile.vllm -t $(VLLM_FULL_IMAGE) .

.PHONY: vllm-run
vllm-run: ## Run vLLM server with batch-invariant ops
	docker run --rm --gpus all \
		-p $(VLLM_PORT):8000 \
		-v ~/.cache/huggingface:/root/.cache/huggingface \
		--shm-size 16g \
		-e MODEL_NAME=$(VLLM_MODEL) \
		$(VLLM_FULL_IMAGE)

.PHONY: vllm-run-detached
vllm-run-detached: ## Run vLLM server in detached mode
	docker run -d --name vllm-deterministic --gpus all \
		-p $(VLLM_PORT):8000 \
		-v ~/.cache/huggingface:/root/.cache/huggingface \
		--shm-size 16g \
		-e MODEL_NAME=$(VLLM_MODEL) \
		$(VLLM_FULL_IMAGE)

.PHONY: vllm-compose-up
vllm-compose-up: ## Start vLLM with docker-compose
	docker-compose -f vllm/docker-compose.yml up --build

.PHONY: vllm-compose-up-detached
vllm-compose-up-detached: ## Start vLLM with docker-compose (detached)
	docker-compose -f vllm/docker-compose.yml up -d --build

.PHONY: vllm-compose-down
vllm-compose-down: ## Stop vLLM docker-compose services
	docker-compose -f vllm/docker-compose.yml down

.PHONY: vllm-compose-logs
vllm-compose-logs: ## View vLLM docker-compose logs
	docker-compose -f vllm/docker-compose.yml logs -f

.PHONY: vllm-test
vllm-test: ## Test vLLM determinism (requires running server)
	python vllm/test_determinism.py --requests 100 --concurrency 10

.PHONY: vllm-stop
vllm-stop: ## Stop running vLLM container
	docker stop vllm-deterministic || true
	docker rm vllm-deterministic || true

.PHONY: vllm-logs
vllm-logs: ## Show logs from vLLM container
	docker logs -f vllm-deterministic

.PHONY: vllm-shell
vllm-shell: ## Open shell in vLLM container
	docker exec -it vllm-deterministic /bin/bash

.PHONY: vllm-push
vllm-push: ## Push vLLM image to registry
	docker push $(VLLM_FULL_IMAGE)

.PHONY: vllm-info
vllm-info: ## Show vLLM configuration
	@echo "vLLM Image:  $(VLLM_FULL_IMAGE)"
	@echo "Model:       $(VLLM_MODEL)"
	@echo "Port:        $(VLLM_PORT)"
	@echo "Registry:    $(REGISTRY)"
	@echo "Username:    $(USERNAME)"

