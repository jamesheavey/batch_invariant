# Test vLLM inference with batch-invariant operations for deterministic results
# This ensures that the same input produces identical outputs regardless of batch composition
# Requires: PyTorch 2.9.0+, vLLM with batch invariant ops support (PR #24583)

import asyncio

import httpx
import modal

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.11.0",
        "huggingface_hub[hf_transfer]==0.35.0",
        "flashinfer-python==0.3.1",
        "torch==2.9.0",  # Upgraded to 2.9.0+ for batch invariant ops support
        "triton",  # Required by batch_invariant_ops
        "httpx",
    )
    .add_local_dir(
        "/Users/jamesheavey/batch_invariant_ops/batch_invariant_ops",
        "/root/batch_invariant_ops",
    )
    .add_local_file(
        "/Users/jamesheavey/batch_invariant_ops/pyproject.toml",
        "/root/pyproject.toml",
    )
    .run_commands("pip install -e /root")  # Install batch_invariant_ops as editable package
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)

MODEL_NAME = "Qwen/Qwen3-8B-FP8"
MODEL_REVISION = (
    "220b46e3b2180893580a4454f21f22d3ebb187d3"  # avoid nasty surprises when repos update!
)


hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# We'll also cache some of vLLM's JIT compilation artifacts in a Modal Volume.

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("example-vllm-inference")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=32)  # how many requests can one replica handle? tune carefully!
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import os
    import stat
    import subprocess
    import tempfile
    import textwrap

    # Create a wrapper script that enables batch invariant mode before starting vLLM
    # This ensures deterministic inference across different batch sizes
    # Requires PyTorch 2.9.0+ and vLLM with batch invariant ops support
    # (https://github.com/vllm-project/vllm/pull/24583)
    wrapper_script = textwrap.dedent(
        f"""
        #!/usr/bin/env python3
        import os
        import sys
        
        # Enable batch invariant mode globally
        from batch_invariant_ops import enable_batch_invariant_mode
        enable_batch_invariant_mode()
        print("Batch invariant mode enabled for deterministic inference")
        
        # Now start vLLM with batch invariant ops active
        os.execvp("vllm", sys.argv[1:])
    """
    )

    # Write wrapper script to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(wrapper_script)
        wrapper_path = f.name

    # Make the wrapper executable
    os.chmod(wrapper_path, os.stat(wrapper_path).st_mode | stat.S_IEXEC)

    cmd = [
        "python3",
        wrapper_path,
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "llm",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]

    # enforce-eager disables both Torch compilation and CUDA graph capture
    # default is no-enforce-eager. see the --compilation-config flag for tighter control
    cmd += ["--enforce-eager"]

    # assume multiple GPUs are for splitting up large matrix multiplications
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    print(cmd)

    subprocess.Popen(cmd)


@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES):
    url = serve.get_web_url()

    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "model": "llm",
        "messages": [
            {
                "role": "user",
                "content": "Generate 1000 random numbers. Go directly into it, don't say Sure and don't say here are numbers. Just start with a number. /no_think",
            }
        ],
        "chat_template_kwargs": {"thinking": False},
        "temperature": 0.0,
        "max_tokens": 100,
    }

    print(f"Running health check for server at {url}")
    async with httpx.AsyncClient(base_url=url, timeout=test_timeout) as client:
        health_resp = await client.get("/health")
        assert health_resp.status_code == 200, f"Failed health check for server at {url}"
        print(f"Successful health check for server at {url}")

        print(f"Sending 1000 requests to test deterministic behavior...")

        outs = []
        responses = []
        for i in range(1000):
            response = client.post("/v1/chat/completions", headers=headers, json=data, timeout=120)
            responses.append(response)

        responses = await asyncio.gather(*responses)
        for response in responses:
            outs.append(response.json()["choices"][0]["message"]["content"])

        for i in outs:
            print(i.replace("\n", " "))
        print(f"Total samples: {len(outs)}, Unique samples: {len(set(outs))}")
