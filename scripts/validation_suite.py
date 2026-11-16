"""Fixed, non-adaptive validation: matched baseline + two additional seeds."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from fedbrain.runner import RunConfig,read_json,run_training

base=read_json(ROOT/'configs'/'fedbrain.json')
experiments=[
    ('full-fedavg-s42',dict(method='fedavg',seed=42,feature_run_id='full-fedbrain-s42')),
    ('full-fedbrain-s43',dict(method='fedbrain',seed=43)),
    ('full-fedbrain-s44',dict(method='fedbrain',seed=44)),
]
for run_id,overrides in experiments:
    existing=read_json(ROOT/'runs'/run_id/'result.json',{})
    if existing.get('status')=='completed':
        print('Already completed',run_id,flush=True)
        continue
    result=run_training(RunConfig(**{**base,**overrides}),run_id)
    print('VALIDATION RESULT',run_id,result['metrics'],flush=True)
