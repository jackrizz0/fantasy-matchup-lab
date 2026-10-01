"""Loads the whole app headlessly, like a visitor opening the page, and fails on any error shown on it.

Run:  python tests/test_app_smoke.py   (downloads data on first run, like the app)
"""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_loads_without_errors():
    at = AppTest.from_file(str(APP), default_timeout=900).run()
    errors = [e.value for e in at.exception] + [e.value for e in at.error]
    assert not errors, f"The app showed {len(errors)} error(s):\n" + "\n\n".join(map(str, errors))
    tabs = [t.label for t in at.tabs]
    for name in ("Rankings", "Usage Shifts", "Game Breakdown", "Model Scorecard"):
        assert any(name in t for t in tabs), f"Missing tab: {name} (found {tabs})"
    return tabs


if __name__ == "__main__":
    tabs = test_app_loads_without_errors()
    print(f"App loaded without errors. Tabs: {', '.join(tabs)}")
