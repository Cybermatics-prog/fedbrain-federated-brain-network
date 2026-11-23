# SPDX-License-Identifier: MIT
"""Independent UI check: moving packets never move or change local health charts."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'artifacts'
report={'errors':[],'posts':[],'model_inference':[]}
with sync_playwright() as p:
    browser=p.chromium.launch(channel='msedge',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1080})
    page.on('pageerror',lambda error:report['errors'].append(str(error)))
    def record(request):
        if request.method=='POST': report['posts'].append(request.url)
        if '/api/subjects/' in request.url: report['model_inference'].append(request.url)
    page.on('request',record)
    page.goto('http://127.0.0.1:8765/#overview',wait_until='networkidle')
    page.wait_for_function("document.querySelector('#data-protection-root')?.dataset.ready==='true' && document.querySelector('#data-protection-root')?.dataset.previewsReady==='true'")
    report['real_site_fc_count']=page.locator('.df-fc').count()
    report['real_site_bold_count']=page.locator('.df-bold polyline').count()
    report['aggregate_coordinates']=page.locator('#df-server .df-param-cell').count()
    report['server_raw_charts']=page.locator('#df-server canvas,#df-server .df-bold').count()
    page.screenshot(path=str(OUT/'dataflow_lead_first.png'))
    page.locator('.dataflow-card').screenshot(path=str(OUT/'dataflow_lead_card.png'))
    before=page.locator('.df-fc').evaluate_all('(nodes)=>nodes.map(n=>({id:n.id,pixels:n.toDataURL(),x:n.getBoundingClientRect().x,y:n.getBoundingClientRect().y}))')
    page.locator('#df-play').click()
    page.wait_for_function("Array.from(document.querySelectorAll('.df-packet')).some(n=>Number(getComputedStyle(n).opacity)>.1)")
    report['animation_visible_packets']=page.locator('.df-packet').evaluate_all('(nodes)=>nodes.filter(n=>Number(getComputedStyle(n).opacity)>.1).length')
    report['packet_types']=page.locator('.df-packet').evaluate_all('(nodes)=>[...new Set(nodes.map(n=>n.dataset.packetKind))]')
    during=page.locator('.df-fc').evaluate_all('(nodes)=>nodes.map(n=>({id:n.id,pixels:n.toDataURL(),x:n.getBoundingClientRect().x,y:n.getBoundingClientRect().y}))')
    report['local_pixels_and_positions_unchanged']=before==during
    page.locator('.dataflow-card').screenshot(path=str(OUT/'dataflow_lead_motion.png'))
    report['overview_still_active']=page.locator('#view-overview.active').is_visible()
    browser.close()
(OUT/'dataflow_invariance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
assert report['real_site_fc_count']==6 and report['real_site_bold_count']==6
assert report['aggregate_coordinates']==64 and report['server_raw_charts']==0
assert report['local_pixels_and_positions_unchanged'] and report['overview_still_active']
assert report['animation_visible_packets']>0
assert set(report['packet_types'])=={'parameter','statistics','global'}
assert not report['errors'] and not report['posts'] and not report['model_inference']
