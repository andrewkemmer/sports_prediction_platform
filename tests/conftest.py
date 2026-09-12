"""Shared pytest configuration: put the project root on sys.path and block network."""

import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _block_network_access(monkeypatch):
    """Guardrail: tests must never access the real network (hardened rule 7/16)."""
    def _blocked(*args, **kwargs):
        raise AssertionError("Network access blocked in tests by autouse socket guard (rule 7/16)")

    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
