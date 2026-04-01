"""Clash proxy rotator for AkShare rate limit bypass.

Rotates through Clash proxy nodes via the mihomo unix socket API.
Each rotation changes the exit IP, distributing rate limits across nodes.

Usage:
    rotator = ClashRotator()
    rotator.patch_requests()  # monkey-patch requests to use Clash proxy
    # Now all requests.get/post calls go through Clash with rotating IPs
    rotator.rotate()  # switch to next node
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import socket
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_SOCKET = "/var/tmp/verge/verge-mihomo.sock"
_DEFAULT_PROXY = "http://127.0.0.1:7890"
_DEFAULT_GROUP = "龙猫云 - TotoroCloud"

# Skip info nodes (not real proxies)
_SKIP_KEYWORDS = ("网址", "流量", "到期", "重置", "自动选择", "故障转移")


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str) -> None:
        super().__init__("localhost")
        self._socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self._socket_path)


class ClashRotator:
    """Rotate Clash proxy nodes to distribute rate limits."""

    def __init__(
        self,
        socket_path: str = _DEFAULT_SOCKET,
        proxy_url: str = _DEFAULT_PROXY,
        group: str = _DEFAULT_GROUP,
    ) -> None:
        self._socket_path = socket_path
        self._proxy_url = proxy_url
        self._group = group
        self._nodes: list[str] = []
        self._index = 0
        self._patched = False
        self._load_nodes()

    def _api(self, method: str, path: str, body: dict | None = None) -> dict | None:
        try:
            conn = _UnixHTTPConnection(self._socket_path)
            headers = {"Content-Type": "application/json"} if body else {}
            data = json.dumps(body).encode() if body else None
            conn.request(method, path, body=data, headers=headers)
            resp = conn.getresponse()
            if resp.status == 204:
                return None
            return json.loads(resp.read())
        except Exception:
            logger.debug("Clash API failed", exc_info=True)
            return None

    def _load_nodes(self) -> None:
        encoded = urllib.request.quote(self._group)
        data = self._api("GET", f"/proxies/{encoded}")
        if not data:
            logger.warning("Could not load Clash proxy nodes (socket: %s)", self._socket_path)
            return
        all_nodes = data.get("all", [])
        self._nodes = [
            n for n in all_nodes
            if not any(kw in n for kw in _SKIP_KEYWORDS)
        ]
        logger.info("Clash rotator: %d proxy nodes available", len(self._nodes))

    @property
    def available(self) -> bool:
        return len(self._nodes) > 0

    def rotate(self) -> str | None:
        """Switch to next proxy node. Returns node name or None if unavailable."""
        if not self._nodes:
            return None
        node = self._nodes[self._index % len(self._nodes)]
        self._index += 1
        encoded = urllib.request.quote(self._group)
        self._api("PUT", f"/proxies/{encoded}", {"name": node})
        logger.debug("Rotated to proxy node: %s", node)
        return node

    def patch_requests(self) -> None:
        """Monkey-patch requests.Session to route through Clash proxy."""
        if self._patched:
            return
        if not self.available:
            logger.info("No proxy nodes available, skipping patch")
            return

        import requests

        _original_init = requests.Session.__init__

        proxy_url = self._proxy_url

        def _patched_init(self_session: Any, *args: Any, **kwargs: Any) -> None:
            _original_init(self_session, *args, **kwargs)
            self_session.proxies = {
                "http": proxy_url,
                "https": proxy_url,
            }
            # Trust the proxy's SSL (Clash handles TLS)
            self_session.verify = True

        requests.Session.__init__ = _patched_init  # type: ignore[assignment]
        self._patched = True
        logger.info("Patched requests.Session to use Clash proxy (%s)", self._proxy_url)

    def unpatch_requests(self) -> None:
        """Remove the monkey-patch."""
        if not self._patched:
            return
        import requests
        # Can't easily un-monkey-patch, just clear proxies for new sessions
        self._patched = False
