"""Independently recompute every completed run's metrics and leakage checks."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from fedbrain.runner import atomic_json,infer_subject,read_json
from fedbrain.metrics import grouped_metrics


def main():
    checks=[]
    validation_only_checks=[]
    for directory in sorted((ROOT/'runs').iterdir()):
        if not directory.is_dir():
            continue
        result=read_json(directory/'result.json',{})
        if result.get('status')!='completed':
            continue
        split=read_json(directory/'split.json')
        sets={k:set(str(s['id']) for s in v) for k,v in split.items()}
        assert not sets['train']&sets['val'] and not sets['train']&sets['test'] and not sets['val']&sets['test']
        if result.get('metrics') is None:
            assert result['config'].get('defer_test') and not result.get('test_evaluated')
            assert not (directory/'predictions.json').exists()
            validation=read_json(directory/'validation_predictions.json')
            assert set(str(p['id']) for p in validation)==sets['val']
            recomputed=grouped_metrics(validation)
            assert np.isclose(recomputed['site_mean_accuracy'],result['validation_metrics']['site_mean_accuracy'])
            assert result['selected_round']<=result['round']==result['config']['rounds']
            validation_only_checks.append({'id':result['id'],'passed':True,'test_not_evaluated':True,
                                           'validation_accuracy':recomputed['site_mean_accuracy']})
            continue
        predictions=read_json(directory/'predictions.json')
        assert set(str(p['id']) for p in predictions)==sets['test']
        assert len(predictions)==len(sets['test'])
        recomputed=grouped_metrics(predictions)
        for key in ('accuracy','balanced_accuracy','auc','sensitivity','specificity','site_mean_accuracy'):
            assert np.isclose(recomputed[key],result['metrics'][key]),(directory,key)
        assert recomputed['confusion_matrix']==result['metrics']['confusion_matrix']
        assert result['round']==result['config']['rounds']==len(result['history'])
        assert np.isfinite([h['train_loss'] for h in result['history']]).all()
        assert all(a['communication_bytes']<=b['communication_bytes'] for a,b in zip(result['history'],result['history'][1:]))
        transform=np.load(directory/'preprocessing.npz')
        ix=transform['indices']
        assert len(np.unique(ix))==len(ix) and ix.min()>=0 and ix.max()<6670
        assert transform['mean'].shape==transform['std'].shape==(6670,)
        assert (transform['std']>0).all()
        reloaded=infer_subject(directory,predictions[0]['id'])
        reload_error=abs(reloaded['prediction']['asd_probability']-predictions[0]['asd_probability'])
        assert reload_error<1e-5,(directory,'checkpoint replay mismatch',reload_error)
        assert len(reloaded['fc'])==116 and len(reloaded['fc'][0])==116
        checks.append({'id':result['id'],'passed':True,'n_test':len(predictions),'n_features':len(ix),
                       'checkpoint_reload_probability_error':reload_error,
                       'macro_site_accuracy':recomputed['site_mean_accuracy'],
                       'pooled_accuracy':recomputed['accuracy']})
    encrypted=read_json(ROOT/'runs'/'encrypted-smoke-2048'/'predictions.json')
    plain=read_json(ROOT/'runs'/'smoke-six-sites-v2'/'predictions.json')
    encrypted_error=None
    if encrypted and plain:
        ep={r['id']:r for r in encrypted}
        assert set(ep)==set(r['id'] for r in plain)
        encrypted_error=max(abs(ep[r['id']]['asd_probability']-r['asd_probability']) for r in plain)
        assert encrypted_error<1e-4,encrypted_error
    power=read_json(ROOT/'artifacts'/'real_power_gate.json',{})
    assert power.get('passed') and power['all_subjects_from_train_only']
    evidence={'checks':checks,'validation_only_checks':validation_only_checks,'all_completed_runs_passed':True,
              'encrypted_vs_plain_max_probability_error':encrypted_error,
              'real_data_power_gate_passed':True}
    atomic_json(ROOT/'artifacts'/'audit_results.json',evidence)
    print(evidence)


if __name__=='__main__':
    main()
