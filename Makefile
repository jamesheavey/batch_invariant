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

