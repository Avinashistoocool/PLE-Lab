"""Headless smoke test of the explorer UI: clicks every tab, fails on console errors, saves screenshots.

    python -m explorer.screenshot_tour --url http://127.0.0.1:8765 [--out explorer/screenshots]
Requires a running server and playwright (uses the installed Edge: channel='msedge').
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def tour(url: str, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.goto(url)
        page.wait_for_selector("#nb-table tr.link", timeout=60_000)

        def idle() -> None:
            page.wait_for_function("document.getElementById('loading').classList.contains('hidden')", timeout=60_000)
            page.wait_for_timeout(300)

        def visible_banner() -> None:
            b = page.locator("#banner")
            if b.is_visible() and "warn" not in (b.get_attribute("class") or ""):
                errors.append(f"error banner: {b.inner_text()}")

        idle()
        page.screenshot(path=out / "token.png", full_page=True)
        page.locator("#nb-table tr.link").first.click()
        idle(); visible_banner()

        page.click("button[data-tab=map]"); idle()
        page.wait_for_selector("#map-plot canvas")
        page.mouse.move(700, 450)
        page.screenshot(path=out / "map.png", full_page=True); visible_banner()

        page.click("button[data-tab=analyze]"); idle()
        page.wait_for_selector("#an-chips .chip")
        page.locator("#an-chips .chip").nth(3).hover()
        page.screenshot(path=out / "analyze.png", full_page=True)
        page.check("#an-ablate"); idle()
        page.wait_for_selector("#an-ablation-summary table")
        page.locator("#an-chips .chip").nth(5).click()
        page.screenshot(path=out / "analyze_ablate.png", full_page=True); visible_banner()

        page.click("button[data-tab=stats]"); idle()
        page.wait_for_selector("#stats-hist canvas")
        page.screenshot(path=out / "stats.png", full_page=True); visible_banner()

        page.click("button[data-tab=compare]"); idle()
        page.screenshot(path=out / "compare.png", full_page=True); visible_banner()

        page.click("button[data-tab=about]"); idle()
        page.screenshot(path=out / "about.png", full_page=True)
        browser.close()
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--out", default="explorer/screenshots")
    args = ap.parse_args()
    errors = tour(args.url, Path(args.out))
    print("\n".join(errors) if errors else "no console errors")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
