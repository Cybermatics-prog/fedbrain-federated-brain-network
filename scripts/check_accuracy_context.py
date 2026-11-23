# SPDX-License-Identifier: MIT
"""GET-only browser guard: never confuse process-demo with full-run metrics."""
import json
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser=p.chromium.launch(channel='msedge',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1080})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8765',wait_until='networkidle')
    page.wait_for_function("document.getElementById('aux-run-context').textContent.includes('8 rounds')")
    demo=page.locator('#aux-run-context').inner_text()
    assert 'Not the 50-round main experiment' in demo
    page.locator('#run-select').select_option('full-fedbrain-s42',force=True)
    page.wait_for_function("document.getElementById('aux-run-context').textContent.includes('50 rounds') && !document.getElementById('aux-run-context').textContent.includes('Not the 50-round main experiment')")
    original=page.locator('#aux-run-context').inner_text()
    assert not errors,errors
    print(json.dumps({'demo_context':demo,'main_context':original,'js_errors':errors},ensure_ascii=False))
    browser.close()
