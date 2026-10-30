"""
app/make_screenshots.py
=======================
Headless automated screenshot capture for presentation decks.
Checks for playwright availability; prints fallback notice if not installed.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    try:
        pw_sync = __import__("playwright.sync_api", fromlist=["sync_playwright"])
        sync_playwright = pw_sync.sync_playwright
    except (ImportError, ModuleNotFoundError):
        print("screenshots skipped (playwright not installed)")
        return

    # Start Streamlit server as subprocess
    port = 8599
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "app" / "streamlit_app.py"),
        "--server.headless",
        "true",
        "--server.port",
        str(port),
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3)

    pages = [
        "System Overview",
        "Live Triage",
        "Batch Analysis",
        "Alert Log",
        "Results",
        "Downloads",
    ]

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            url = f"http://localhost:{port}"

            for page_name in pages:
                slug = page_name.lower().replace(" ", "_")
                page.goto(url)
                page.wait_for_load_state("networkidle")

                # Select navigation radio if not on first page
                if page_name != "System Overview":
                    page.get_by_label(page_name).check()
                    page.wait_for_load_state("networkidle")

                out_path = FIGURES_DIR / f"system_{slug}.png"
                page.screenshot(path=str(out_path), full_page=True)
                print(f"Captured screenshot: {out_path.name}")

            browser.close()
    except Exception as ex:
        print(f"Screenshot capture encountered error: {ex}")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()
