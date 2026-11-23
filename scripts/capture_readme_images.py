# SPDX-License-Identifier: MIT
"""Capture the README hero image from the live demo (no data modification)."""
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / 'docs' / 'images'
IMAGES.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(channel='msedge', headless=True)
    page = browser.new_page(viewport={'width': 1440, 'height': 1080}, device_scale_factor=2)
    page.goto('http://127.0.0.1:8765/#overview', wait_until='networkidle', timeout=90000)
    page.wait_for_function(
        "document.querySelector('#data-protection-root')?.dataset.ready==='true'"
        " && document.querySelector('#data-protection-root')?.dataset.previewsReady==='true'",
        timeout=90000)
    page.wait_for_timeout(600)
    page.locator('.dataflow-card').screenshot(path=str(IMAGES / 'federation-data-flow.png'))
    page.locator('#health-root .health-controls').screenshot(path=str(IMAGES / 'health-controls.png'))
    page.screenshot(path=str(IMAGES / 'overview.png'), full_page=True)
    page.locator('.nav-item[data-view="protocol"]').click()
    page.wait_for_timeout(900)
    page.screenshot(path=str(IMAGES / 'protocol.png'), full_page=True)
    page.set_viewport_size({'width': 390, 'height': 844})
    page.goto('http://127.0.0.1:8765/#overview', wait_until='networkidle')
    page.wait_for_function("document.querySelector('#data-protection-root')?.dataset.previewsReady==='true'", timeout=60000)
    page.locator('.dataflow-card').screenshot(path=str(IMAGES / 'federation-data-flow-mobile.png'))
    browser.close()
for image in sorted(IMAGES.glob('*.png')):
    print(image.name, round(image.stat().st_size / 1024, 1), 'KB')
