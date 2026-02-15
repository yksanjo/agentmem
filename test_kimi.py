#!/usr/bin/env python3
"""Quick test of Kimi API integration."""

import asyncio
import os

import httpx
from dotenv import load_dotenv

# Load env
load_dotenv()


async def test_kimi():
    api_key = os.environ.get("KIMI_API_KEY")
    if not api_key:
        print("ERROR: KIMI_API_KEY not set")
        return False

    print(f"API Key: {api_key[:10]}...{api_key[-4:]}")

    async with httpx.AsyncClient(
        base_url="https://api.moonshot.cn/v1",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=60.0,
    ) as client:
        try:
            print("Testing Kimi API...")
            response = await client.post(
                "/chat/completions",
                json={
                    "model": "moonshot-v1-8k",
                    "messages": [{"role": "user", "content": "Say 'Hello!' in one word."}],
                    "max_tokens": 50,
                },
            )
            response.raise_for_status()
            data = response.json()
            result = data["choices"][0]["message"]["content"]
            print(f"Response: {result}")
            print("Kimi integration working!")
            return True
        except httpx.HTTPStatusError as e:
            print(f"HTTP Error: {e.response.status_code}")
            print(f"Details: {e.response.text}")
            return False
        except Exception as e:
            print(f"Error: {e}")
            return False


if __name__ == "__main__":
    success = asyncio.run(test_kimi())
    exit(0 if success else 1)
