"""Minimal copy of P1's orders + courier mock API (FastAPI), on Lumi Skin synthetic data.

Endpoints (same paths as P1): GET /orders/{order_id} and GET /tracking/{tracking_id}.
Run it on its own with:  uvicorn voice_agent.mock_api:app --port 8001
"""

import csv
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, HTTPException

from voice_agent.config import DATA_DIR


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def normalise_order_id(order_id: str) -> str:
    """Accept 'LS-10045', 'ls10045' or '10045' and return 'LS-10045'."""
    digits = "".join(ch for ch in order_id if ch.isdigit())
    return f"LS-{digits}"


def load_shop(data_dir: Path = DATA_DIR) -> dict:
    """Load orders, customer phones and tracking events into dictionaries."""
    orders = {row["order_id"]: row for row in _read_csv(data_dir / "orders.csv")}
    phones = {row["id"]: row["phone"] for row in _read_csv(data_dir / "customers.csv")}
    events = defaultdict(list)
    for row in _read_csv(data_dir / "tracking_events.csv"):
        events[row["tracking_id"]].append(row)
    for rows in events.values():
        rows.sort(key=lambda r: r["timestamp"])
    return {"orders": orders, "phones": phones, "events": dict(events)}


def create_app(data_dir: Path = DATA_DIR) -> FastAPI:
    shop = load_shop(data_dir)
    app = FastAPI(title="Lumi Skin orders + courier mock API")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "orders": len(shop["orders"])}

    @app.get("/orders/{order_id}")
    def get_order(order_id: str) -> dict:
        order = shop["orders"].get(normalise_order_id(order_id))
        if order is None:
            raise HTTPException(status_code=404, detail="order not found")
        phone = shop["phones"].get(order["customer_id"], "")
        # Data minimisation: the agent only needs the last 4 phone digits to verify
        # the caller, so the API never returns the full phone, name or address.
        last4 = "".join(ch for ch in phone if ch.isdigit())[-4:]
        return {
            "order_id": order["order_id"],
            "status": order["status"],
            "order_date": order["order_date"],
            "shipped_date": order["shipped_date"] or None,
            "delivered_date": order["delivered_date"] or None,
            "tracking_id": order["tracking_id"] or None,
            "destination_country": order["destination_country"],
            "customer_phone_last4": last4,
        }

    @app.get("/tracking/{tracking_id}")
    def get_tracking(tracking_id: str) -> dict:
        events = shop["events"].get(tracking_id)
        if events is None:
            raise HTTPException(status_code=404, detail="tracking id not found")
        return {
            "tracking_id": tracking_id,
            "events": [
                {"timestamp": e["timestamp"], "event": e["event"], "location": e["location"]}
                for e in events
            ],
        }

    return app


app = create_app()
