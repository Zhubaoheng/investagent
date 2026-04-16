#!/usr/bin/env python3
"""Quick LLM profile sanity check.

Usage:
    uv run python scripts/llm_diag.py              # test all configured profiles
    uv run python scripts/llm_diag.py claude       # test only specified profiles

Reports each profile's status (ok / error), model name, and single-call latency.
Safe to run during production pipelines; uses a minimal 16-token probe.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from poorcharlie.config import create_llm_client  # noqa: E402
from poorcharlie.llm_profiles import (  # noqa: E402
    list_available_profiles,
    resolve_default_profile,
)


async def ping(profile: str) -> dict:
    try:
        llm = create_llm_client(profile=profile)
    except Exception as e:
        return {"profile": profile, "ok": False, "error": f"config: {type(e).__name__}: {e}"}

    t0 = time.time()
    try:
        response = await llm.create_message(
            system="Respond with exactly one word: ok",
            messages=[{"role": "user", "content": "Are you alive?"}],
            tools=[],
            max_tokens=16,
        )
        elapsed = round(time.time() - t0, 2)
        content_preview = ""
        for block in getattr(response, "content", []):
            if getattr(block, "type", None) == "text":
                content_preview = (block.text or "")[:40]
                break
        return {
            "profile": profile,
            "ok": True,
            "latency_s": elapsed,
            "model": llm.model,
            "provider": llm.provider,
            "reply": content_preview,
        }
    except Exception as e:
        return {
            "profile": profile,
            "ok": False,
            "latency_s": round(time.time() - t0, 2),
            "error": f"{type(e).__name__}: {e}",
        }


async def ping_generic() -> dict:
    """Test the generic LLM_BASE_URL / LLM_API_KEY / LLM_MODEL path (no profile name needed)."""
    import os
    base = os.getenv("LLM_BASE_URL")
    key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    if not (base and key and model):
        return {"ok": False, "error": "LLM_BASE_URL / LLM_API_KEY / LLM_MODEL not all set"}
    try:
        llm = create_llm_client(base_url=base, api_key=key, model=model,
                                provider=os.getenv("LLM_PROVIDER", "openai"))
    except Exception as e:
        return {"ok": False, "error": f"config: {type(e).__name__}: {e}"}
    t0 = time.time()
    try:
        response = await llm.create_message(
            system="Respond with exactly one word: ok",
            messages=[{"role": "user", "content": "Are you alive?"}],
            tools=[], max_tokens=16,
        )
        elapsed = round(time.time() - t0, 2)
        reply = ""
        for block in getattr(response, "content", []):
            if getattr(block, "type", None) == "text":
                reply = (block.text or "")[:40]
                break
        return {"ok": True, "latency_s": elapsed, "model": llm.model,
                "provider": llm.provider, "reply": reply}
    except Exception as e:
        return {"ok": False, "latency_s": round(time.time() - t0, 2),
                "error": f"{type(e).__name__}: {e}"}


async def main() -> None:
    args = [p for p in sys.argv[1:] if not p.startswith("-")]
    profiles = args if args else list_available_profiles()
    default = resolve_default_profile()

    print(f"Default profile: {default}")
    print(f"Configured profiles: {list_available_profiles()}")

    # Always test the generic LLM_* path (most users configure this way)
    import os
    has_generic = all(os.getenv(k) for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"))
    if has_generic:
        result = await ping_generic()
        status = "✓" if result["ok"] else "✗"
        if result["ok"]:
            print(
                f"{status} {'LLM_*':10s}  {result['latency_s']:>5.2f}s  "
                f"{result['model']} ({result['provider']})  reply={result['reply']!r}"
            )
        else:
            print(f"{status} {'LLM_*':10s}  ERROR: {result['error']}")

    if not profiles and not has_generic:
        print("(no config found — set LLM_BASE_URL + LLM_API_KEY + LLM_MODEL in .env)")
        return

    if profiles:
        print(f"\nNamed profiles: {profiles}")
        for p in profiles:
            result = await ping(p)
            status = "✓" if result["ok"] else "✗"
            if result["ok"]:
                print(
                    f"{status} {p:10s}  {result['latency_s']:>5.2f}s  "
                    f"{result['model']} ({result['provider']})  reply={result['reply']!r}"
                )
            else:
                print(f"{status} {p:10s}  ERROR: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())
