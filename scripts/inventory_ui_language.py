"""Collect user-facing text from recorded APIs without changing source records."""
import json,re,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'artifacts'/'english'
OUT.mkdir(parents=True,exist_ok=True)
han=re.compile(r'[\u3400-\u9fff]')
texts={}
def collect(value,path):
    if isinstance(value,dict):
        for k,v in value.items(): collect(v,path+'/'+k)
    elif isinstance(value,list):
        for i,v in enumerate(value): collect(v,path+'/'+str(i))
    elif isinstance(value,str) and han.search(value):
        texts.setdefault(value,[]).append(path)
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for url in ['/api/status','/api/optimization','/api/data-protection','/api/health-data',
            '/api/health-data/sites','/api/health-data/subjects/51036',
            '/api/runs/federation-demo-s42','/api/runs/federation-demo-s42/federation',
            '/api/runs/encrypted-smoke-2048/federation']:
    with opener.open('http://127.0.0.1:8765'+url,timeout=45) as response:
        collect(json.load(response),url)
(OUT/'api-text.json').write_text(json.dumps(texts,ensure_ascii=False,indent=2),encoding='utf-8')
source=(ROOT/'web'/'federation.js').read_text(encoding='utf-8')
fragments=list(dict.fromkeys(re.findall(r'[\u3400-\u9fff][\u3400-\u9fff\s，；：、。！？（）「」·→/]*',source)))
(OUT/'federation-text.json').write_text(json.dumps(fragments,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'unique_api_texts':len(texts),'federation_fragments':len(fragments)},ensure_ascii=False))
