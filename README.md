# Batch Invariant Ops

A companion library release to https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/. This library contains some batch-invariant kernels as well as an example of achieving deterministic vLLM inference.

## Overview

This library primarily leverages torch.Library to sub out existing PyTorch kernels with "batch-invariant" ones. This allows many existing PyTorch models to use the batch-invariant ops with low overhead and non-intrusive code changes.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import torch
from batch_invariant_ops import set_batch_invariant_mode

# Enable batch-invariant mode
with set_batch_invariant_mode():
    # Your inference code here
    model = YourModel()
    output = model(input_tensor)
```

## Testing Batch-Invariance

### Low-Level Matrix Operations Test

The following example shows how batch size can affect results in standard PyTorch:

```python
import torch
from batch_invariant_ops import set_batch_invariant_mode
torch.set_default_device('cuda')

# Just to get the logging out of the way haha
with set_batch_invariant_mode(True):
    pass

def test_batch_invariance():
    B, D = 2048, 4096
    a = torch.linspace(-100, 100, B*D).reshape(B, D)
    b = torch.linspace(-100, 100, D*D).reshape(D, D)

    # Method 1: Matrix-vector multiplication (batch size 1)
    out1 = torch.mm(a[:1], b)

    # Method 2: Matrix-matrix multiplication, then slice (full batch)
    out2 = torch.mm(a, b)[:1]

    # Check if results are identical
    diff = (out1 - out2).abs().max()
    print(f"Difference: {diff.item()}")
    return diff.item() == 0

# Test with standard PyTorch (likely to show differences)
print("Standard PyTorch:")
with set_batch_invariant_mode(False):
    is_deterministic = test_batch_invariance()
    print(f"Deterministic: {is_deterministic}")

# Test with batch-invariant operations
print("\nBatch-Invariant Mode:")
with set_batch_invariant_mode(True):
    is_deterministic = test_batch_invariance()
    print(f"Deterministic: {is_deterministic}")

```

### Model Generation Test

Test batch invariance with actual language model generations:

```bash
python test_model_generation.py
# or
make test-model
```

This script demonstrates the core principle of batch invariance: **a given input should always produce the same output, regardless of what else is in the batch**.

The test generates text for the same prompt using greedy decoding across multiple different batch contexts:

1. Solo generation (batch_size=1)
2. With short padding prompts (batch_size=3)
3. With medium padding prompts (batch_size=3)
4. With long padding prompts (batch_size=3)
5. Large batch with varied lengths (batch_size=5)

**In standard PyTorch:** Different batch compositions trigger different CUDA kernels and cause different floating-point accumulation patterns, leading to slightly different logits. With greedy decoding, these small differences can cause completely different token selections, resulting in different outputs for the _same input_.

**With batch-invariant operations:** The same input produces identical outputs across all batch contexts, ensuring true determinism.

## Deterministic Inference in vLLM

`deterministic_vllm_inference.py` shows an proof of concept of validating that vLLM can be made deterministic with a minor upstream PR to use this library. Without the upstream PR, we see that out of 1000 random length 100 completions we see 18 unique samples. After the upstream PR, there is only one unique sample.

## Supported Operations

### Matrix Operations

- `torch.mm()` - Matrix multiplication
- `torch.addmm()` - Matrix multiplication with bias addition

### Activation Functions

- `torch.log_softmax()` - Log-softmax activation

### Reduction Operations

- `torch.mean()` - Mean computation along specified dimensions
