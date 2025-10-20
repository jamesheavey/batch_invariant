#!/usr/bin/env python3
"""
Test script to verify deterministic outputs from vLLM with batch-invariant ops.

This script hammers the vLLM server with concurrent requests and verifies that
all outputs are byte-identical when using temperature=0.

Usage:
    python test_determinism.py --url http://localhost:8000 --requests 100 --concurrency 10
"""

import argparse
import asyncio
import sys
from typing import List

import httpx


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 100,
) -> str:
    """Send a single completion request to the vLLM server."""
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": 1.0,
        "n": 1,
        "max_tokens": max_tokens,
    }

    response = await client.post(
        url,
        json=data,
        timeout=120.0,
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]


async def test_determinism(
    url: str,
    model: str,
    prompt: str,
    num_requests: int = 100,
    concurrency: int = 10,
    temperature: float = 0.0,
    max_tokens: int = 100,
) -> None:
    """
    Test determinism by sending multiple concurrent requests and checking
    that all outputs are identical.
    """
    print(f"Testing determinism with {num_requests} requests ({concurrency} concurrent)")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Prompt: {prompt[:50]}...")
    print("-" * 80)

    outputs: List[str] = []

    async with httpx.AsyncClient() as client:
        # Send requests in batches to control concurrency
        for batch_start in range(0, num_requests, concurrency):
            batch_size = min(concurrency, num_requests - batch_start)
            print(
                f"Sending batch {batch_start // concurrency + 1} ({batch_size} requests)...",
                end=" ",
            )

            tasks = [
                send_request(client, url, model, prompt, temperature, max_tokens)
                for _ in range(batch_size)
            ]

            batch_outputs = await asyncio.gather(*tasks)
            outputs.extend(batch_outputs)
            print("✓")

    print("-" * 80)
    print(f"\nResults:")
    print(f"  Total requests: {len(outputs)}")

    unique_outputs = list(set(outputs))
    print(f"  Unique outputs: {len(unique_outputs)}")

    if len(unique_outputs) == 1:
        print("\n✓ SUCCESS: All outputs are identical (deterministic)!")
        print(f"\nOutput:\n{unique_outputs[0]}")
        return 0
    else:
        print("\n✗ FAILURE: Outputs are non-deterministic")
        print(f"\nFound {len(unique_outputs)} unique outputs:")
        for i, output in enumerate(unique_outputs[:5], 1):  # Show first 5
            print(f"\n--- Output {i} (appeared {outputs.count(output)} times) ---")
            print(output)
            if i == 5 and len(unique_outputs) > 5:
                print(f"\n... and {len(unique_outputs) - 5} more unique outputs")
        return 1


async def check_health(base_url: str) -> bool:
    """Check if the vLLM server is healthy."""
    health_url = f"{base_url.rstrip('/v1').rstrip('/v1/chat/completions')}/health"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(health_url, timeout=5.0)
            return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test vLLM determinism with batch-invariant ops")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/v1/chat/completions",
        help="vLLM server URL (default: http://localhost:8000/v1/chat/completions)",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name to use",
    )
    parser.add_argument(
        "--prompt",
        default="Explain batch invariance in one sentence.",
        help="Prompt to test with",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Number of requests to send (default: 100)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent requests (default: 10)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)",
    )

    args = parser.parse_args()

    # Check server health first
    print("Checking server health...")
    if not asyncio.run(check_health(args.url)):
        print("ERROR: Server is not responding. Make sure vLLM is running.")
        return 1
    print("Server is healthy ✓\n")

    # Run determinism test
    result = asyncio.run(
        test_determinism(
            args.url,
            args.model,
            args.prompt,
            args.requests,
            args.concurrency,
            args.temperature,
            args.max_tokens,
        )
    )

    return result


if __name__ == "__main__":
    sys.exit(main())
