# SPDX-License-Identifier: MIT
"""Verify real health plots still work when experiment APIs are unavailable."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent.parent
with sync_playwright() as p:
    browser=p.chromium.launch(channel='msedge',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1080})
    errors=[]
    posts=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('request',lambda r:posts.append(r.url) if r.method=='POST' else None)
    def unavailable(route):
        route.fulfill(status=503,content_type='application/json',body='{"detail":"Experiment API intentionally unavailable in browser-only test"}')
    page.route('**/api/status',unavailable)
    page.route('**/api/runs',unavailable)
    page.route('**/api/runs/**',unavailable)
    page.route('**/api/optimization',unavailable)
    page.goto('http://127.0.0.1:8765',wait_until='networkidle')
    page.wait_for_selector('#health-root[data-ready="true"]')
    page.wait_for_function("document.getElementById('health-root').dataset.subjectId !== ''")
    checks={'classification_apis_unavailable':True,
            'healthy_data_ready':page.locator('#health-root').get_attribute('data-ready'),
            'counts':page.locator('#health-counts strong').all_text_contents(),
            'real_roi_nodes':page.locator('#health-network [data-health-roi]').count(),
            'real_signal_series':page.locator('#health-signal-chart polyline').count(),
            'selected_real_subject':page.locator('#health-root').get_attribute('data-subject-id'),
            'no_federation_in_health_view':page.locator('#view-overview #federation-root').count()==0,
            'js_errors':errors,'posts':posts}
    assert checks['counts']==['465','465','241','224'],checks
    assert checks['real_roi_nodes']==116 and checks['real_signal_series']==3,checks
    assert not errors and not posts,checks
    normal=browser.new_page(viewport={'width':1440,'height':1080})
    normal.goto('http://127.0.0.1:8765',wait_until='networkidle')
    normal.wait_for_function("document.getElementById('health-root').dataset.subjectId !== ''")
    inference_requests=[]
    normal.on('request',lambda r:inference_requests.append(r.url) if '/api/subjects/' in r.url else None)
    normal.locator('[data-health-group="HC"]').click()
    normal.locator('#health-network [data-health-roi="10"] circle').click()
    normal.locator('.health-modes [data-health-mode="reference"]').click()
    normal.wait_for_load_state('networkidle')
    assert not inference_requests,inference_requests
    checks['normal_health_clicks_trigger_no_model_inference']=True
    checks['unexpected_inference_requests']=inference_requests
    (ROOT/'artifacts'/'health_frontend_independence.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False,indent=2))
    browser.close()
