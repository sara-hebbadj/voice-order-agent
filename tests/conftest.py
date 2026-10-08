import csv

import pytest

from voice_agent.agent import OrderStatusAgent
from voice_agent.config import DATA_DIR
from voice_agent.tools import make_order_tools


@pytest.fixture(scope="session")
def tools():
    return make_order_tools()  # in-process mock API, no network


@pytest.fixture
def agent(tools):
    return OrderStatusAgent(tools)


@pytest.fixture(scope="session")
def orders():
    """order_id -> order row plus the customer's phone last 4 digits."""
    with (DATA_DIR / "customers.csv").open(encoding="utf-8") as f:
        last4 = {c["id"]: c["phone"][-4:] for c in csv.DictReader(f)}
    with (DATA_DIR / "orders.csv").open(encoding="utf-8") as f:
        rows = {o["order_id"]: o for o in csv.DictReader(f)}
    for row in rows.values():
        row["phone4"] = last4[row["customer_id"]]
    return rows


def talk(agent, *lines):
    """Send several caller lines and return the list of agent turns."""
    return [agent.handle(line) for line in lines]
