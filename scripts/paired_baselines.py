# SPDX-License-Identifier: MIT
"""Fixed matched-mask baseline seeds43/44; no score-adaptive configuration."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from fedbrain.runner import RunConfig,read_json,run_training

base=read_json(ROOT/'configs'/'fedbrain.json')
for seed in (43,44):
    source=f'full-fedbrain-s{seed}'
    prior=read_json(ROOT/'runs'/source/'result.json',{})
    if prior.get('status')!='completed':
        raise SystemExit(f'Wait for {source} to complete before starting paired baseline')
    run_id=f'full-fedavg-s{seed}'
    existing=read_json(ROOT/'runs'/run_id/'result.json',{})
    if existing.get('status')=='completed':
        continue
    result=run_training(RunConfig(**{**base,'seed':seed,'method':'fedavg','feature_run_id':source}),run_id)
    print('PAIRED BASELINE',run_id,result['metrics'],flush=True)
