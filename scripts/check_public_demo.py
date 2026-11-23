# SPDX-License-Identifier: MIT
"""Check the published demo and prove its write endpoints are fenced."""
import argparse
import json
from pathlib import Path
import urllib.error
import urllib.request
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser()
parser.add_argument('--url',required=True)
args=parser.parse_args()
base=args.url.rstrip('/')
if not base.startswith('https://') or not base.endswith('.trycloudflare.com'):
    raise ValueError('Expected the generated HTTPS Cloudflare Quick Tunnel URL')

def request(path,body=None):
    payload=None if body is None else json.dumps(body).encode()
    req=urllib.request.Request(base+path,data=payload,headers={'Content-Type':'application/json','User-Agent':'FedBrain-demo-verification'})
    try:
        response=urllib.request.urlopen(req,timeout=60)
    except urllib.error.HTTPError as error:
        response=error
    with response:
        return response.status,response.read()

report={'url':base,'browser_errors':[],'checks':{}}
code,body=request('/api/health')
report['checks']['public_health_ok']=code==200 and json.loads(body).get('ok') is True
code,body=request('/api/health-data/sites')
sites=json.loads(body)
report['checks']['six_real_site_previews']=code==200 and len(sites.get('sites',[]))==6
report['checks']['public_subject_count']=sites.get('total_subjects')
# Invalid training input could never launch a job even if a wrong origin were exposed.
code,_=request('/api/train',{'rounds':0})
report['checks']['training_post_blocked']=code==403
code,_=request('/api/runs/no-such-public-test-run/stop',{})
report['checks']['stop_post_blocked']=code==403
code,_=request('/ui_dataflow_smoke.py')
report['checks']['python_source_not_public']=code==404
code,_=request('/checkpoint.pt')
report['checks']['model_files_not_public']=code==404
with sync_playwright() as p:
    browser=p.chromium.launch(channel='msedge',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1080})
    page.on('pageerror',lambda error:report['browser_errors'].append(str(error)))
    response=page.goto(base+'/#overview',wait_until='networkidle',timeout=90000)
    page.wait_for_function("document.querySelector('#data-protection-root')?.dataset.ready==='true' && document.querySelector('#data-protection-root')?.dataset.previewsReady==='true'",timeout=90000)
    report['checks']['browser_http_ok']=response.status==200
    report['checks']['browser_site_fc']=page.locator('.df-fc').count()
    report['checks']['browser_site_bold']=page.locator('.df-bold polyline').count()
    report['checks']['browser_aggregate_parameters']=page.locator('#df-server .df-param-cell').count()
    report['checks']['health_page_active']=page.locator('#view-overview.active').is_visible()
    page.screenshot(path=str(ROOT/'artifacts'/'public_access'/'public-browser.png'))
    browser.close()
(ROOT/'artifacts'/'public_access'/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
assert not report['browser_errors']
for key in ('public_health_ok','six_real_site_previews','training_post_blocked','stop_post_blocked','python_source_not_public','model_files_not_public','browser_http_ok','health_page_active'):
    assert report['checks'][key],key
assert report['checks']['public_subject_count']==465
assert report['checks']['browser_site_fc']==report['checks']['browser_site_bold']==6
assert report['checks']['browser_aggregate_parameters']==64
