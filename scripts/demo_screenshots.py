"""Screenshot every demo tab for the slides: reports/figures/demo_<tab>.png.

    pip install playwright && python -m playwright install chromium   # once (dev tool only)
    python scripts/demo_screenshots.py                # BigQuery data (falls back to local)
    VSE_DATA=local python scripts/demo_screenshots.py # local files only

Starts the app headless on a free port, opens it in headless Chromium, clicks each tab and saves
a full-page screenshot, then stops the app.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
CONTENT_HEIGHT_JS = """() => Math.max(...[...document.querySelectorAll(
  '[data-testid="stMain"], [data-testid="stAppViewContainer"], section.main, [data-testid="stMainBlockContainer"]'
)].map(e => e.scrollHeight), document.body.scrollHeight)"""
TABS = ["Overview", "Recall watchlist", "AI agent", "Model results", "Trajectory", "Data and limitations"]
# (tab, key of the button that opens the view, key of its back button, file name)
DETAILS = [("AI agent", "agent_open_2024-12-r06", "agent_back", "demo_agent_detail"),
           ("Model results", "open_results_hit", "back_results", "demo_detail_hit_rate")]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(url: str, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/_stcore/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"streamlit did not start at {url}")


def shoot(page, args, file_name: str) -> None:
    page.set_viewport_size({"width": args.width, "height": args.height})
    page.wait_for_timeout(2500)  # charts finish drawing
    # Streamlit scrolls an inner container, so grow the window to the content height
    height = page.evaluate(CONTENT_HEIGHT_JS)
    page.set_viewport_size({"width": args.width, "height": max(args.height, height + 40)})
    page.mouse.move(2, 2)  # park the mouse so no hover tooltip is captured
    page.wait_for_timeout(1500)
    path = OUT / f"{file_name}.png"
    page.screenshot(path=str(path))
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright

    port = free_port()
    url = f"http://127.0.0.1:{port}"
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.headless=true",
         f"--server.port={port}", "--browser.gatherUsageStats=false", "--theme.base=light"],
        cwd=ROOT, env=os.environ.copy(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        wait_for(url)
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": args.width, "height": args.height},
                                    device_scale_factor=2)
            page.goto(url, wait_until="networkidle")
            page.get_by_role("tab", name="Overview").wait_for(timeout=120_000)  # data loaded
            for name in TABS:
                page.get_by_role("tab", name=name).click()
                shoot(page, args, f"demo_{name.lower().replace(' ', '_')}")
            for tab, open_key, back_key, file_name in DETAILS:
                page.get_by_role("tab", name=tab).click()
                page.wait_for_timeout(1000)
                page.locator(f".st-key-{open_key} button").first.click()
                shoot(page, args, file_name)
                page.locator(f".st-key-{back_key} button").click()
                page.wait_for_timeout(1000)
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
