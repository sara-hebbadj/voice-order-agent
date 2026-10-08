"""The agent's two tools: look up an order and look up its courier tracking.

They call the orders + courier API over HTTP. By default the mock API runs in-process
(no server to start); set ORDER_API_URL to call a separately running server instead,
for example P1's mock API.
"""

import time
from dataclasses import dataclass, field

import httpx


@dataclass
class ToolCall:
    """One tool call, kept for the demo UI and the evaluation traces."""

    name: str
    args: dict
    ok: bool
    latency_ms: float
    result: dict | None = field(default=None, repr=False)


class OrderTools:
    def __init__(self, http: httpx.Client):
        self.http = http

    def _get(self, name: str, path: str, args: dict) -> ToolCall:
        start = time.perf_counter()
        try:
            response = self.http.get(path)
            ok = response.status_code == 200
            result = response.json() if ok else None
        except httpx.HTTPError:
            ok, result = False, None
        latency_ms = (time.perf_counter() - start) * 1000
        return ToolCall(name=name, args=args, ok=ok, latency_ms=latency_ms, result=result)

    def get_order(self, order_id: str) -> ToolCall:
        return self._get("get_order", f"/orders/{order_id}", {"order_id": order_id})

    def get_tracking(self, tracking_id: str) -> ToolCall:
        return self._get("get_tracking", f"/tracking/{tracking_id}", {"tracking_id": tracking_id})


def make_order_tools(order_api_url: str | None = None) -> OrderTools:
    """Use a real HTTP server if a URL is given, otherwise the in-process mock API."""
    if order_api_url:
        return OrderTools(httpx.Client(base_url=order_api_url, timeout=5.0))
    # Starlette's TestClient sends real HTTP requests to the app without a network socket.
    from fastapi.testclient import TestClient

    from voice_agent.mock_api import create_app

    return OrderTools(TestClient(create_app()))
