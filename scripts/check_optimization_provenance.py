"""Read-only scientific provenance checks; does not evaluate any subject."""
import hashlib
import sys
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from fedbrain.runner import RunConfig,atomic_json,read_json
from optimize import SELECTION,VARIANTS,configuration


def main():
    selection=read_json(SELECTION)
    if not selection:
        raise RuntimeError('Selection not yet persisted')
    checks=[]
    for variant in VARIANTS:
        p=ROOT/'runs'/f'opt-{variant}-s42'
        r=read_json(p/'result.json')
        expected=asdict(RunConfig(**configuration(variant,42,True)))
        assert r['config']==expected,(variant,'configuration mismatch')
        checkpoint=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert checkpoint['config']==expected
        assert checkpoint['selected_round']==r['selected_round']
        hashes={name:hashlib.sha256((ROOT/'fedbrain'/name).read_bytes()).hexdigest() for name in r['source_hashes']}
        assert hashes==r['source_hashes'],(variant,'source modified after fit')
        if not r.get('test_evaluated'):
            assert r['metrics'] is None and not (p/'predictions.json').exists()
        checks.append({'id':r['id'],'config_equal':True,'source_hashes_equal':True,
                       'test_evaluated_at_snapshot':r.get('test_evaluated',False),
                       'checkpoint_sha256':hashlib.sha256((p/'checkpoint.pt').read_bytes()).hexdigest(),
                       'checkpoint_mtime':datetime.fromtimestamp((p/'checkpoint.pt').stat().st_mtime,timezone.utc).isoformat()})
    selected=next(v for v in selection['variants'] if v['id']==selection['selected_variant'])
    expected=max([v for v in selection['variants'] if v['status']=='completed'],key=lambda v:v['validation_accuracy'])
    assert selected['id']==expected['id']
    assert selected['config']==selection['selected_config']
    evidence={'captured_at':datetime.now(timezone.utc).isoformat(),
              'selection_locked_at':selection['locked_at'],
              'selection_sha256':hashlib.sha256(SELECTION.read_bytes()).hexdigest(),
              'winner_is_validation_maximum':True,'all_requested_configs_match':True,
              'all_training_sources_match':True,'checks':checks,
              'not_a_tamper_proof_security_guarantee':True}
    atomic_json(ROOT/'artifacts'/'optimization_provenance.json',evidence)
    print(evidence)


if __name__=='__main__':
    main()
