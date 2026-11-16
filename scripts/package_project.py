"""Package source, reproducibility evidence and trained models, not cached data."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import zipfile
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from fedbrain.runner import atomic_json


def main():
    paths=[]
    roots=['fedbrain','configs','scripts','tests','web','docs','runs']
    for root in roots:
        for p in (ROOT/root).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or '.pytest_cache' in p.parts:
                continue
            if p.name=='paper.txt' or p.suffix in ('.tmp','.part','.log'):
                continue
            paths.append(p)
    paths.extend(ROOT/p for p in ['README.md','requirements.txt','setup.ps1','start.ps1','.gitignore'])
    paths.extend(p for p in (ROOT/'artifacts').glob('*.json') if p.name!='package_manifest.json')
    distributions={}
    for package in ('numpy','scipy','scikit-learn','torch','fastapi','uvicorn','requests','pydantic'):
        distributions[package]=importlib.metadata.version(package)
    manifest={'created_at':datetime.now(timezone.utc).isoformat(),
              'python':sys.version,'platform':platform.platform(),'dependencies':distributions,
              'excluded':'raw ABIDE derivatives and FC caches, .venv, full paper extraction',
              'download_command':'python -m fedbrain.runner download',
              'files':{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}}
    atomic_json(ROOT/'artifacts'/'package_manifest.json',manifest)
    paths.append(ROOT/'artifacts'/'package_manifest.json')
    name='fedbrain-demo-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'.zip'
    output=ROOT/'artifacts'/name
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(set(paths)):
            archive.write(path,arcname='fedbrain-demo/'+str(path.relative_to(ROOT)).replace('\\','/'))
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
    print(json.dumps({'archive':str(output),'bytes':output.stat().st_size,'file_count':len(set(paths))},ensure_ascii=False))


if __name__=='__main__':
    main()
