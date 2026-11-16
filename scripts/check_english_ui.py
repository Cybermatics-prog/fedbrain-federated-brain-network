"""Verify the English interface, its data-flow behavior, and the public page."""
import argparse,json,re
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'artifacts'/'english'
OUT.mkdir(parents=True,exist_ok=True)
parser=argparse.ArgumentParser()
parser.add_argument('--url',default='http://127.0.0.1:8765')
parser.add_argument('--public',action='store_true')
args=parser.parse_args()
base=args.url.rstrip('/')
report={'url':base,'errors':[],'unexpected_posts':[],'checks':{},'language_issues':{}}
SCAN=r'''() => {
 const han=/[\u3400-\u9fff]/, text=[];
 const walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
 while(walker.nextNode()) {
   const n=walker.currentNode;
   if(!['SCRIPT','STYLE'].includes(n.parentElement?.tagName) && han.test(n.textContent)) text.push(n.textContent.trim().slice(0,300));
 }
 const attributes=[];
 document.querySelectorAll('*').forEach(n=>['aria-label','aria-description','title','placeholder','alt'].forEach(k=>{
   const v=n.getAttribute(k);if(v&&han.test(v))attributes.push({tag:n.tagName,id:n.id,key:k,value:v});
 }));
 return {text:[...new Set(text)],attributes};
}'''
def snapshot(page,label):
    page.evaluate("window.scrollTo({top:0,behavior:'instant'})")
    page.evaluate("() => Promise.all(Array.from(document.querySelectorAll('.view.active')).flatMap(n=>n.getAnimations()).map(a=>a.finished.catch(()=>null)))")
    issues=page.evaluate(SCAN)
    report['language_issues'][label]=issues
    report['checks'][label+'_no_horizontal_overflow']=page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
    page.screenshot(path=str(OUT/(label+'.png')),full_page=True)
    return issues
with sync_playwright() as p:
    browser=p.chromium.launch(channel='msedge',headless=True)
    context=browser.new_context(viewport={'width':1440,'height':1080})
    page=context.new_page()
    page.on('pageerror',lambda error:report['errors'].append(str(error)))
    page.on('request',lambda request:report['unexpected_posts'].append(request.url) if request.method=='POST' else None)
    page.goto(base+'/?english-check=1#overview',wait_until='networkidle',timeout=90000)
    page.wait_for_function("document.querySelector('#health-root')?.dataset.ready==='true' && document.querySelector('#data-protection-root')?.dataset.previewsReady==='true'",timeout=90000)
    report['checks']['html_language']=page.locator('html').get_attribute('lang')
    report['checks']['local_fc_count']=page.locator('.df-fc').count()
    report['checks']['local_bold_count']=page.locator('.df-bold polyline').count()
    report['checks']['aggregate_coordinates']=page.locator('#df-server .df-param-cell').count()
    page.locator('.df-details').evaluate('(n)=>n.open=true')
    snapshot(page,'public-overview' if args.public else 'overview')
    if not args.public:
        page.locator('[data-df-mode="upload"]').click()
        snapshot(page,'upload-payloads')
        report['checks']['uploaded_coordinates']=page.locator('.df-upload-grid .df-param-cell').count()
        page.locator('[data-df-mode="local"]').click()
        page.locator('#df-play').click()
        page.wait_for_function("Array.from(document.querySelectorAll('.df-packet')).some(n=>Number(getComputedStyle(n).opacity)>.1)")
        report['checks']['packet_types']=page.locator('.df-packet').evaluate_all('(nodes)=>[...new Set(nodes.map(n=>n.dataset.packetKind))]')
        snapshot(page,'transfer')
        page.locator('[data-health-group="difference"]').click()
        snapshot(page,'group-difference')
        page.locator('[data-health-mode="reference"]').first.click()
        snapshot(page,'hc-reference')
        page.locator('#health-age-min').fill('90')
        page.locator('#health-age-max').fill('10')
        page.locator('#health-apply').click()
        snapshot(page,'invalid-age-message')
        page.locator('#health-age-min').fill('')
        page.locator('#health-age-max').fill('')
        for view in ['train','inference','protocol']:
            page.locator(f'.nav-item[data-view="{view}"]').click()
            page.wait_for_function(f"document.querySelector('#view-{view}')?.classList.contains('active')")
            if view=='train':
                page.locator('.training-method-record').evaluate('(n)=>n.open=true')
            if view=='inference':
                page.wait_for_function("document.querySelectorAll('#subject-list [data-subject-id]').length>0 || document.querySelector('#subject-notice')?.textContent.length>0",timeout=60000)
            snapshot(page,view)
        page.locator('.nav-item[data-view="overview"]').click()
        page.locator('#protection-run').select_option('encrypted-smoke-2048')
        page.wait_for_function("document.querySelector('#data-protection-root')?.dataset.run==='encrypted-smoke-2048'")
        snapshot(page,'historical-encryption')
        report['checks']['legacy_no_invented_coordinates']=page.locator('.df-param-cell').count()==0
        page.set_viewport_size({'width':390,'height':844})
        page.goto(base+'/?english-check=mobile#overview',wait_until='networkidle')
        page.wait_for_function("document.querySelector('#data-protection-root')?.dataset.previewsReady==='true'",timeout=60000)
        snapshot(page,'mobile-overview')
        # A separate context exercises a Chinese API error without changing the service.
        failure=browser.new_page(viewport={'width':1440,'height':1080})
        failure.route('**/api/health-data*',lambda route:route.fulfill(status=503,content_type='application/json',body=json.dumps({'detail':'健康数据接口暂不可用'},ensure_ascii=False)))
        failure.goto(base+'/?english-check=error#overview',wait_until='networkidle')
        snapshot(failure,'error-state')
        failure.close()
    report['checks']['untranslated_api_notes']=page.evaluate('Array.from(window.EnglishUI?.untranslated || [])')
    browser.close()
(OUT/('public-report.json' if args.public else 'report.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
assert not report['errors'],report['errors']
assert not report['unexpected_posts'],report['unexpected_posts']
assert report['checks']['html_language']=='en'
assert report['checks']['local_fc_count']==report['checks']['local_bold_count']==6
assert report['checks']['aggregate_coordinates']==64
for key,value in report['checks'].items():
    if key.endswith('_no_horizontal_overflow'): assert value,key
for key,issues in report['language_issues'].items():
    assert not issues['text'] and not issues['attributes'],(key,issues)
