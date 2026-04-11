"""Standalone PDF extraction worker — runs as a subprocess to bypass GIL.

Protocol: reads raw PDF bytes from stdin, writes JSON result to stdout.
Input:  {"action": "extract_markdown"|"extract_sections", "market": "...", ...}
        followed by raw PDF bytes (length specified in header).
Output: JSON with {"text": "..."} or {"sections": {...}} or {"error": "..."}

Usage:
    echo '<header_json>\n<pdf_bytes>' | python -m poorcharlie.datasources.pdf_extract_worker
"""

from __future__ import annotations

import json
import sys


def _extract_markdown(raw_content: bytes) -> str:
    from poorcharlie.datasources.pdf_extract import extract_pdf_markdown
    return extract_pdf_markdown(raw_content)


def _extract_sections(text: str, market: str) -> dict[str, str]:
    from poorcharlie.datasources.pdf_extract import extract_sections
    return extract_sections(text, market)


def main() -> None:
    """Read a single task from stdin, execute, write result to stdout."""
    # Read header line (JSON)
    header_line = sys.stdin.buffer.readline()
    if not header_line:
        sys.exit(0)

    try:
        header = json.loads(header_line)
    except json.JSONDecodeError as e:
        sys.stdout.write(json.dumps({"error": f"Invalid header: {e}"}))
        sys.stdout.flush()
        sys.exit(1)

    action = header.get("action", "")
    data_len = header.get("data_len", 0)

    # Read binary data if specified
    raw_data = b""
    if data_len > 0:
        raw_data = sys.stdin.buffer.read(data_len)

    try:
        if action == "extract_markdown":
            text = _extract_markdown(raw_data)
            result = {"text": text}
        elif action == "extract_sections":
            text = header.get("text", "")
            market = header.get("market", "")
            result = {"sections": _extract_sections(text, market)}
        else:
            result = {"error": f"Unknown action: {action}"}
    except Exception as e:
        result = {"error": str(e)}

    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
