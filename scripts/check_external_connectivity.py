"""
Quick TCP connectivity check for external AI dependencies.

Usage:
  python scripts/check_external_connectivity.py
"""

from __future__ import annotations

import socket
from typing import Iterable, Tuple


def check_tcp(host: str, port: int = 443, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def run_checks(targets: Iterable[Tuple[str, int]]) -> int:
    failed = 0
    print("External Connectivity Check")
    print("=" * 36)
    for host, port in targets:
        ok = check_tcp(host, port)
        status = "OK" if ok else "BLOCKED"
        print(f"- {host}:{port} -> {status}")
        if not ok:
            failed += 1
    return failed


def main() -> None:
    targets = [
        ("api.pinecone.io", 443),
        ("api.openai.com", 443),
    ]
    failed = run_checks(targets)
    if failed:
        print("\nResult: outbound network issue detected.")
    else:
        print("\nResult: all dependency endpoints reachable.")


if __name__ == "__main__":
    main()
