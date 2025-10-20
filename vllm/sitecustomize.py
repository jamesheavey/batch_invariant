"""
sitecustomize.py - Auto-enable batch-invariant kernels for deterministic inference

This module is automatically imported by Python on startup (when on PYTHONPATH).
It enables batch-invariant mode for all vLLM workers, ensuring deterministic
outputs at temperature=0 by replacing non-deterministic PyTorch operations
with batch-invariant implementations.

The batch-invariant operations ensure that:
- Matrix multiplications (mm, addmm) use fixed reduction order
- Softmax operations are stable across batch sizes
- Mean reductions are batch-shape independent

This eliminates the subtle FP differences that cause divergent argmax selections
in greedy decoding when dynamic batching is used.
"""

import atexit
import sys

try:
    from batch_invariant_ops import set_batch_invariant_mode

    # Enter batch-invariant mode globally
    _ctx = set_batch_invariant_mode(True)
    _ctx.__enter__()

    # Ensure proper cleanup on exit
    atexit.register(_ctx.__exit__, None, None, None)

    # Optional: print confirmation (useful for debugging)
    # Uncomment the line below if you want to see confirmation in logs
    # print("[sitecustomize] Batch-invariant mode enabled", file=sys.stderr)

except ImportError as e:
    print(f"[sitecustomize] WARNING: Could not import batch_invariant_ops: {e}", file=sys.stderr)
    print("[sitecustomize] Deterministic execution will NOT be enabled", file=sys.stderr)
except Exception as e:
    print(f"[sitecustomize] ERROR enabling batch-invariant mode: {e}", file=sys.stderr)
    raise
