"""Playwright check for explorer/static/viz-gallery.html (dev harness, not part of the test suite).

python explorer/viz_gallery_check.py --url http://127.0.0.1:8773 --out explorer/screenshots
"""
import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8773")
    ap.add_argument("--out", default="explorer/screenshots")
    ap.add_argument("--reduced", action="store_true", help="emulate prefers-reduced-motion")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        pg = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1,
                        reduced_motion="reduce" if args.reduced else "no-preference")
        pg.on("console", lambda m: m.type == "error" and "Failed to load resource" not in m.text and errors.append(f"console: {m.text}"))
        pg.on("response", lambda r: r.status >= 400 and "favicon" not in r.url and errors.append(f"HTTP {r.status} {r.url}"))
        pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        pg.goto(f"{args.url}/static/viz-gallery.html")
        pg.wait_for_function("document.body.dataset.ready", timeout=120_000)
        if pg.evaluate("document.body.dataset.ready") != "1":
            errors.append("gallery: " + pg.inner_text("#status"))
        pg.wait_for_timeout(2500)

        def shot(sel: str, name: str) -> None:
            pg.locator(sel).screenshot(path=str(out / f"viz-{name}.png"))

        def hover(sel: str, fx: float, fy: float) -> None:
            box = pg.locator(sel).bounding_box()
            pg.mouse.move(box["x"] + box["width"] * fx, box["y"] + box["height"] * fy, steps=4)
            pg.wait_for_timeout(250)

        # token tab
        hover("#glyph canvas", 0.72, 0.30)
        shot("#tab-token", "token")
        hover("#heat canvas", 0.4, 0.3)
        shot("#heat", "heatmap")
        hover("#sim svg", 0.55, 0.6)
        # force graph: drag a non-root node
        nodes = pg.locator("#force g.node")
        n = nodes.count()
        assert n > 3, f"force graph has {n} nodes"
        nodes.nth(3).scroll_into_view_if_needed()
        pg.wait_for_timeout(500)
        nb = nodes.nth(3).bounding_box()
        pg.mouse.move(nb["x"] + nb["width"] / 2, nb["y"] + nb["height"] / 2)
        pg.mouse.down()
        pg.mouse.move(nb["x"] + 140, nb["y"] - 90, steps=12)
        pg.mouse.up()
        pg.wait_for_timeout(1200)
        nb2 = nodes.nth(3).bounding_box()
        moved = abs(nb2["x"] - nb["x"]) + abs(nb2["y"] - nb["y"])
        if moved < 30:
            errors.append(f"drag did not move the node ({moved:.1f}px)")
        hover("#force g.node:nth-child(5)", 0.5, 0.5)
        shot("#force", "force")
        # analyze tab
        pg.locator("#tab-analyze").scroll_into_view_if_needed()
        hover("#river svg", 0.45, 0.4)
        chip_hot = pg.locator("#chips .viz-chip.hot").count()
        if chip_hot != 1:
            errors.append(f"river hover should highlight exactly one chip, got {chip_hot}")
        shot("#tab-analyze", "analyze")
        hover("#chips .viz-chip:nth-child(7)", 0.5, 0.5)
        hover("#abl svg", 0.62, 0.35)
        shot("#abl", "ablation")
        # stats
        pg.locator("#tab-stats").scroll_into_view_if_needed()
        hover("#ridge svg", 0.3, 0.5)
        shot("#tab-stats", "stats")
        hover("#hex canvas", 0.2, 0.75)
        shot("#hex", "hexbin")
        # compare
        pg.locator("#tab-compare").scroll_into_view_if_needed()
        hover("#slope svg:nth-child(2)", 0.5, 0.3)
        shot("#tab-compare", "compare")
        # token switch animation + navigation via force graph click
        pg.locator("#force g.node").nth(1).click()
        pg.wait_for_timeout(1800)
        shot("#tab-token", "token-nav")
        pg.screenshot(path=str(out / "viz-gallery-full.png"), full_page=True)
        b.close()
    for e in errors:
        print("ERROR", e)
    print("ok" if not errors else f"{len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
