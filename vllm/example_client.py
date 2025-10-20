#!/usr/bin/env python3
"""
Example client for vLLM server with batch-invariant ops.

This script demonstrates how to use the deterministic vLLM server from Python.

Usage:
    python vllm/example_client.py
"""

import asyncio

import httpx


# Simple synchronous example
def simple_completion(prompt: str, temperature: float = 0.0) -> str:
    """Send a single completion request to vLLM."""
    response = httpx.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 200,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# Async example for concurrent requests
async def async_completion(prompt: str, temperature: float = 0.0) -> str:
    """Send a single async completion request to vLLM."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 200,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


async def concurrent_completions(prompts: list[str], temperature: float = 0.0) -> list[str]:
    """Send multiple completion requests concurrently."""
    tasks = [async_completion(prompt, temperature) for prompt in prompts]
    return await asyncio.gather(*tasks)


# Streaming example
async def streaming_completion(prompt: str) -> None:
    """Stream tokens from vLLM as they're generated."""
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "stream": True,
            },
            timeout=120.0,
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_lines():
                if chunk.startswith("data: "):
                    data = chunk[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break
                    try:
                        import json

                        json_data = json.loads(data)
                        if "choices" in json_data and len(json_data["choices"]) > 0:
                            delta = json_data["choices"][0].get("delta", {})
                            if "content" in delta:
                                print(delta["content"], end="", flush=True)
                    except json.JSONDecodeError:
                        continue
    print()  # Newline at the end


def main():
    """Run example completions."""
    print("=" * 80)
    print("vLLM Batch-Invariant Ops Example Client")
    print("=" * 80)
    print()

    # Example 1: Simple completion
    print("Example 1: Simple completion")
    print("-" * 80)
    prompt = "Explain what deterministic inference means in one sentence."
    print(f"Prompt: {prompt}")
    print()
    result = simple_completion(prompt)
    print(f"Response: {result}")
    print()

    # Example 2: Test determinism with multiple requests
    print("Example 2: Testing determinism (5 identical requests)")
    print("-" * 80)
    prompt = "What is 2 + 2?"
    print(f"Prompt: {prompt}")
    print()

    results = asyncio.run(concurrent_completions([prompt] * 5))
    unique = set(results)

    print(f"Received {len(results)} responses")
    print(f"Unique responses: {len(unique)}")

    if len(unique) == 1:
        print("✓ All responses are identical (deterministic)!")
        print(f"Response: {results[0]}")
    else:
        print("✗ Responses differ (non-deterministic)")
        for i, response in enumerate(unique, 1):
            print(f"\nResponse {i}: {response}")
    print()

    # Example 3: Different prompts concurrently
    print("Example 3: Multiple different prompts concurrently")
    print("-" * 80)
    prompts = [
        "What is Python?",
        "What is Docker?",
        "What is machine learning?",
    ]

    results = asyncio.run(concurrent_completions(prompts))

    for prompt, result in zip(prompts, results):
        print(f"Q: {prompt}")
        print(f"A: {result}")
        print()

    # Example 4: Streaming
    print("Example 4: Streaming response")
    print("-" * 80)
    prompt = "Count from 1 to 10."
    print(f"Prompt: {prompt}")
    print("Response: ", end="")
    asyncio.run(streaming_completion(prompt))
    print()


if __name__ == "__main__":
    try:
        main()
    except httpx.ConnectError:
        print("ERROR: Could not connect to vLLM server.")
        print("Make sure the server is running: make vllm-compose-up")
        exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        exit(0)
