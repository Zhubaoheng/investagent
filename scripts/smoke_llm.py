"""Smoke test: unified create_llm_client() against whatever .env points at."""
from __future__ import annotations

import asyncio
from dotenv import load_dotenv

load_dotenv()

from poorcharlie.config import create_llm_client, load_llm_config_from_env


async def main() -> None:
    cfg = load_llm_config_from_env("LLM")
    print(f"provider={cfg.provider}  base_url={cfg.base_url}  model={cfg.model}")

    llm = create_llm_client(
        extra_body={"context_window_size": 200000, "effort": "high"}
        if cfg.provider == "minimax" else None,
    )

    resp = await llm.create_message(
        system="You are a test echoer.",
        messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
        tools=[],
        max_tokens=32,
    )
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            print(f"reply: {block.text!r}")
    print(f"stop_reason={resp.stop_reason}  usage={resp.usage}")


if __name__ == "__main__":
    asyncio.run(main())
