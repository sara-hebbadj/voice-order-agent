from fastapi.testclient import TestClient

from voice_agent.mock_api import create_app, normalise_order_id

client = TestClient(create_app())


def test_normalise_order_id():
    assert normalise_order_id("10045") == "LS-10045"
    assert normalise_order_id("ls-10045") == "LS-10045"


def test_get_order_returns_status_and_only_last4_phone_digits(orders):
    response = client.get("/orders/10154")
    assert response.status_code == 200
    body = response.json()
    assert body["order_id"] == "LS-10154"
    assert body["status"] == orders["LS-10154"]["status"]
    assert body["customer_phone_last4"] == orders["LS-10154"]["phone4"]
    # data minimisation: no full phone, name, email or address
    assert not {"phone", "name", "email", "address", "customer_id"} & set(body)


def test_unknown_order_is_404():
    assert client.get("/orders/LS-10999").status_code == 404


def test_tracking_events_are_in_time_order(orders):
    tracking_id = orders["LS-10154"]["tracking_id"]
    events = client.get(f"/tracking/{tracking_id}").json()["events"]
    times = [e["timestamp"] for e in events]
    assert times == sorted(times) and events[0]["event"] == "picked_up"


def test_unknown_tracking_is_404():
    assert client.get("/tracking/TRK0000000").status_code == 404
