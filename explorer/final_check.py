"""End-to-end check of the integrated Explorer shell + viz charts (Playwright, Edge, headless).

    python explorer/final_check.py --url http://127.0.0.1:8774

Visits every tab in dark and light themes, inspects " dog", runs Analyze with an ablation of all layers,
opens Stats and Compare, toggles the map between 2D and 3D, and fails on console errors or HTTP >= 400.
Screenshots: explorer/screenshots/final-<tab>-<theme>.png
"""
import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent / 'screenshots'
TEXT = 'Once upon a time, a happy dog ran to the park.'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:8774')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch(channel='msedge', headless=True, args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        page = b.new_page(viewport={'width': 1440, 'height': 1000})
        page.on('console', lambda m: m.type == 'error' and errors.append(f'console: {m.text}'))
        page.on('pageerror', lambda e: errors.append(f'pageerror: {e}'))
        page.on('response', lambda r: r.status >= 400 and errors.append(f'HTTP {r.status} {r.url}'))

        def idle(ms=900):
            page.wait_for_function('() => document.getElementById("loading").classList.contains("hidden")', timeout=60000)
            page.wait_for_timeout(ms)

        def tab(name):
            page.click(f'#tabs button[data-tab="{name}"]')
            idle()

        def shot(name):
            page.screenshot(path=str(OUT / f'final-{name}.png'), full_page=True)

        def check_cards(name):
            empty = page.evaluate('''() => [...document.querySelectorAll('.tab.active .card:not(.hidden) .chart')]
                .filter(c => c.offsetParent && c.getBoundingClientRect().height < 30).map(c => c.id)''')
            if empty:
                errors.append(f'{name}: empty chart containers {empty}')

        page.goto(args.url)
        idle(1500)
        for theme in ('dark', 'light'):
            if page.evaluate('document.documentElement.dataset.theme || "dark"') != theme:
                page.click('#theme-toggle')
                idle()
            tab('token')
            page.fill('#token-q', ' dog')
            page.press('#token-q', 'Enter')
            idle(2500)
            check_cards(f'token-{theme}')
            shot(f'token-{theme}')

            tab('analyze')
            page.fill('#an-text', TEXT)
            page.evaluate('''() => { const a = document.getElementById('an-all'); a.checked = true;
                const s = document.getElementById('an-ablate'); s.checked = true; s.dispatchEvent(new Event('change')); }''')
            idle(2000)
            page.hover('#an-chips .viz-chip[data-i="4"]')
            page.wait_for_timeout(300)
            if page.evaluate('document.getElementById("an-abl-card").classList.contains("hidden")'):
                errors.append('ablation card hidden after ablation')
            check_cards(f'analyze-{theme}')
            shot(f'analyze-{theme}')

            tab('stats')
            idle(1500)
            check_cards(f'stats-{theme}')
            shot(f'stats-{theme}')

            tab('compare')
            idle(1500)
            shot(f'compare-{theme}')

            tab('map')
            page.evaluate('''() => { const m = document.getElementById('map-mode'); m.value = '2d'; m.dispatchEvent(new Event('change')); }''')
            idle(1500)
            shot(f'map2d-{theme}')
            page.evaluate('''() => { const m = document.getElementById('map-mode'); m.value = '3d'; m.dispatchEvent(new Event('change')); }''')
            idle(4000)
            shot(f'map3d-{theme}')

            tab('about')
            shot(f'about-{theme}')
        b.close()
    errors = [e for e in errors if 'favicon' not in e]
    for e in errors:
        print('ERROR', e)
    print('OK' if not errors else f'{len(errors)} problem(s)')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
