# SPDX-License-Identifier: MIT
"""Bounded, preregistered validation-only search, then locked paired confirmation.

Never computes a candidate's held-out score until the validation decision is
persisted. Existing historical scores are read only after that decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from fedbrain.data import SITE_ORDER,load_subjects
from fedbrain.metrics import grouped_metrics
from fedbrain.model import FedBrainModel
from fedbrain.runner import (RunConfig,_evaluate,_subject_features,atomic_json,
                             read_json,run_training)

VARIANTS={
 'A':{'label':'Validation-set round selection', 'overrides':{}},
 'B':{'label':'Round selection + batch loss scale', 'overrides':{'alignment_weight':1.,'alignment_normalization':'batch'}},
 'C':{'label':'B + refresh alignment target after aggregation', 'overrides':{'alignment_weight':1.,'alignment_normalization':'batch','alignment_reference':'post_aggregation'}},
 'D':{'label':'C + L2 regularization 0.01', 'overrides':{'alignment_weight':1.,'alignment_normalization':'batch','alignment_reference':'post_aggregation','weight_decay':.01}},
 'E':{'label':'C + 400-step sparse optimization', 'overrides':{'alignment_weight':1.,'alignment_normalization':'batch','alignment_reference':'post_aggregation','sparse_max_iter':400,'feature_run_id':None}},
}
ART=ROOT/'artifacts'
SELECTION=ART/'optimization_selection.json'
RESULT=ART/'optimization_results.json'


def configuration(variant,seed,defer_test=True,method='fedbrain'):
    base=read_json(ROOT/'configs'/'fedbrain.json')
    cfg={**base,'seed':seed,'method':method,'checkpoint_selection':'best_val','defer_test':defer_test,
         'feature_run_id':f'full-fedbrain-s{seed}','experiment_tag':f'optimization-{variant}'}
    cfg.update(VARIANTS[variant]['overrides'])
    return cfg


def run_or_reuse(run_id,cfg):
    old=read_json(ROOT/'runs'/run_id/'result.json',{})
    if old.get('status')=='completed':
        if old.get('config')!=asdict(RunConfig(**cfg)):
            raise RuntimeError(f'Completed run {run_id} has a different requested configuration')
        recorded=old.get('source_hashes',{})
        actual={name:hashlib.sha256((ROOT/'fedbrain'/name).read_bytes()).hexdigest() for name in recorded}
        if not recorded or actual!=recorded:
            raise RuntimeError(f'Completed run {run_id} has different or missing training-source fingerprints')
        return old
    if old:
        raise RuntimeError(f'Existing incomplete run {run_id}; keep evidence and use a new tag')
    return run_training(RunConfig(**cfg),run_id)


def search():
    existing=read_json(SELECTION)
    if existing:
        for entry in existing['variants']:
            if asdict(RunConfig(**entry['config']))!=asdict(RunConfig(**configuration(entry['id'],42,True))):
                raise RuntimeError('Locked plan differs; use a new preregistered experiment tag')
        print('Reusing persisted validation decision; no re-search on evaluated candidates',flush=True)
        return existing
    evidence={'status':'searching','selected_variant':None,'selected_run_id':None,'variants':[],
              'changes':['Keep all experts/gates selected on validation','Correct the relative loss scale between classification and domain alignment',
                         'Refresh training alignment statistics after broadcast','Bounded L2/sparse convergence control'],
              'notes':['Candidates are selected only using training/validation data; the test set is not evaluated during the search phase.',
                       'The public cohort, hidden8, noise 0.01 and the 50-round budget are kept unchanged.']}
    atomic_json(RESULT,evidence)
    for variant,definition in VARIANTS.items():
        run_id=f'opt-{variant}-s42'
        cfg=configuration(variant,42,True)
        try:
            result=run_or_reuse(run_id,cfg)
            assert result['metrics'] is None and not result['test_evaluated']
            assert not (ROOT/'runs'/run_id/'predictions.json').exists()
            entry={'id':variant,'run_id':run_id,'label':definition['label'],'status':'completed',
                   'validation_accuracy':result['validation_metrics']['site_mean_accuracy'],
                   'selected_round':result['selected_round'],'test_evaluated':False,
                   'config':cfg,'source_hashes':result.get('source_hashes',{})}
        except Exception as exc:
            entry={'id':variant,'run_id':run_id,'label':definition['label'],'status':'failed',
                   'validation_accuracy':None,'test_evaluated':False,'error':str(exc),'config':cfg}
        evidence['variants'].append(entry)
        atomic_json(RESULT,evidence)
        print('VALIDATION ONLY',json.dumps(entry,ensure_ascii=False),flush=True)
    valid=[v for v in evidence['variants'] if v['status']=='completed']
    if not valid:
        raise RuntimeError('All preregistered candidates failed')
    winner=max(valid,key=lambda v:v['validation_accuracy'])  # Stable order A→E on exact ties.
    baseline=next(v for v in valid if v['id']=='A')
    selection={'locked_at':datetime.now(timezone.utc).isoformat(),'criterion':'seed42 best macro validation; earliest tied checkpoint; fixed candidate order ties',
               'selected_variant':winner['id'],'selected_run_id':winner['run_id'],
               'selected_config':winner['config'],'variants':evidence['variants'],
               'validation_gain_over_A':winner['validation_accuracy']-baseline['validation_accuracy'],
               'validation_minimum_useful_gain':.02,'test_was_not_evaluated_for_selection':True,
               'checkpoint_fingerprints':{v['run_id']:hashlib.sha256((ROOT/'runs'/v['run_id']/'checkpoint.pt').read_bytes()).hexdigest() for v in valid}}
    atomic_json(SELECTION,selection)
    evidence.update(status='confirming',selected_variant=winner['id'],selected_config=winner['config'],
                    validation_gain_over_A=selection['validation_gain_over_A'])
    atomic_json(RESULT,evidence)
    print('LOCKED CONFIGURATION',json.dumps(selection,ensure_ascii=False),flush=True)
    return selection


def finalize_locked_test(run_id,selection):
    persisted=read_json(SELECTION)
    if not persisted or persisted!=selection or not selection['test_was_not_evaluated_for_selection']:
        raise RuntimeError('Exact validation decision must be persisted before opening tests')
    if run_id not in ('opt-A-s42',selection['selected_run_id']):
        raise ValueError('Only baseline and the locked winner may be finalized')
    directory=ROOT/'runs'/run_id
    result=read_json(directory/'result.json')
    if result.get('test_evaluated'):
        return result
    expected_hash=selection.get('checkpoint_fingerprints',{}).get(run_id)
    if not expected_hash or hashlib.sha256((directory/'checkpoint.pt').read_bytes()).hexdigest()!=expected_hash:
        raise RuntimeError('Unfinalized checkpoint lacks the expected pre-test fingerprint')
    checkpoint=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=False)
    planned=next(v['config'] for v in selection['variants'] if v['run_id']==run_id)
    if checkpoint['config']!=asdict(RunConfig(**planned)):
        raise RuntimeError('Checkpoint configuration differs from locked candidate')
    cfg=RunConfig(**checkpoint['config'])
    split=read_json(directory/'split.json')
    lookup={str(s['id']):s for s in load_subjects(cfg.data_dir)}
    subjects=[lookup[str(s['id'])] for s in split['test']]
    ix=checkpoint['indices']
    matrices={}
    for subject in subjects:
        x=_subject_features(subject,cfg)
        matrices[subject['id']]=np.asarray((x[:,ix]-checkpoint['mean'][ix])/checkpoint['std'][ix],dtype=np.float32)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    models={site:FedBrainModel(len(ix),cfg.hidden_dim).to(device) for site in SITE_ORDER}
    for site,model in models.items():
        model.load_state_dict(checkpoint['models'][site])
    records=_evaluate(subjects,matrices,models,device,checkpoint['personalization'])
    atomic_json(directory/'predictions.json',records)
    result['metrics']=grouped_metrics(records)
    result['test_evaluated']=True
    result['test_policy']='One evaluation of validation-locked checkpoint; no test-guided changes'
    result['selection_evidence']='artifacts/optimization_selection.json'
    result['test_unlocked_at']=datetime.now(timezone.utc).isoformat()
    result['generalization_gap']=result['train_metrics']['accuracy']-result['metrics']['accuracy']
    atomic_json(directory/'result.json',result)
    return result


def confirm():
    selection=read_json(SELECTION)
    if not selection:
        raise RuntimeError('Run validation search first')
    variant=selection['selected_variant']
    completed=read_json(RESULT,{})
    if completed.get('status')=='completed':
        if completed.get('selected_config')!=selection['selected_config'] or completed.get('selected_variant')!=variant:
            raise RuntimeError('Completed report does not match persisted selection')
        print('Reusing completed locked confirmation; no additional test evaluation',flush=True)
        return completed
    pairs=[]
    for seed in (42,43,44):
        if seed==42:
            baseline=finalize_locked_test('opt-A-s42',selection)
            optimized=baseline if variant=='A' else finalize_locked_test(selection['selected_run_id'],selection)
        else:
            baseline=run_or_reuse(f'opt-A-s{seed}',configuration('A',seed,False))
            optimized=baseline if variant=='A' else run_or_reuse(f'opt-{variant}-s{seed}',configuration(variant,seed,False))
        pairs.append({'seed':seed,'baseline_id':baseline['id'],'optimized_id':optimized['id'],
                      'baseline':baseline['metrics'],'optimized':optimized['metrics'],
                      'baseline_selected_round':baseline['selected_round'],'optimized_selected_round':optimized['selected_round']})
        print('LOCKED PAIRED CONFIRMATION',json.dumps(pairs[-1],ensure_ascii=False),flush=True)
    fedavg=[]
    for seed in (42,43,44):
        cfg=configuration('A',seed,False,method='fedavg')
        r=run_or_reuse(f'opt-FedAvg-s{seed}',cfg)
        fedavg.append({'id':r['id'],'seed':seed,'metrics':r['metrics'],'selected_round':r['selected_round']})
    old=[read_json(ROOT/'runs'/f'full-fedbrain-s{s}'/'result.json')['metrics'] for s in (42,43,44)]
    differences=np.array([p['optimized']['site_mean_accuracy']-p['baseline']['site_mean_accuracy'] for p in pairs])
    comparison={'seeds':[42,43,44],
                'baseline_macro_mean':float(np.mean([p['baseline']['site_mean_accuracy'] for p in pairs])),
                'optimized_macro_mean':float(np.mean([p['optimized']['site_mean_accuracy'] for p in pairs])),
                'baseline_pooled_mean':float(np.mean([p['baseline']['accuracy'] for p in pairs])),
                'optimized_pooled_mean':float(np.mean([p['optimized']['accuracy'] for p in pairs])),
                'original_macro_mean':float(np.mean([m['site_mean_accuracy'] for m in old])),
                'delta_macro':float(np.mean(differences)),'delta_macro_per_seed':differences.tolist(),
                'success_count':int(np.sum(differences>0)),
                'pass_gate':bool(np.mean(differences)>=.02 and np.sum(differences>0)>=2),
                'fedavg_best_val_macro_mean':float(np.mean([r['metrics']['site_mean_accuracy'] for r in fedavg])),
                'optimized_macro_std':float(np.std([p['optimized']['site_mean_accuracy'] for p in pairs],ddof=1))}
    comparison['paired_delta_95pct_t_interval']=[float(np.mean(differences)+sign*4.3026527299*np.std(differences,ddof=1)/np.sqrt(3)) for sign in (-1,1)]
    result=read_json(RESULT)
    result.update(status='completed',selected_variant=variant,selected_run_id=selection['selected_run_id'],
                  selected_config=selection['selected_config'],comparison=comparison,pairs=pairs,fedavg=fedavg,
                  selection_locked_at=selection['locked_at'],completed_at=datetime.now(timezone.utc).isoformat(),
                  notes=['Candidates are selected only on the seed42 validation set; one evaluation after locking, and no candidate is changed based on the test set.',
                         'The original test set was made public in the previous round; this round is exploratory follow-up validation, not a fresh external validation.',
                         'Rounds are selected on validation within 50 rounds, and all private heads/gates are restored along with the model; the comparison baseline uses the same round-selection rule.',
                         'All candidates and failures are retained; the best seed is not cherry-picked.'])
    atomic_json(RESULT,result)
    config=selection['selected_config'].copy()
    config.update(defer_test=False,seed=42,feature_run_id=None,experiment_tag='optimized-reusable')
    atomic_json(ROOT/'configs'/'optimized.json',config)
    report(result)
    print('FINAL COMPARISON',json.dumps(comparison,ensure_ascii=False),flush=True)
    return result


def report(result):
    comp=result['comparison']
    verdict=('The preregistered minimum-effect gate was passed, but independent external validation is still required.' if comp['pass_gate'] else
             'No stable accuracy improvement was obtained in this round: the preregistered >=2 percentage point minimum-effect gate was not met, so the original default model is not replaced.')
    lines=['# FedBrain Accuracy Optimization Study', '',verdict,'',
           '## Validation Locked First, Test Evaluated Afterwards', '',
           f"Selection time: {result['selection_locked_at']}; variant **{result['selected_variant']}** was selected by seed42 validation macro, not by test performance.", '',
           '|Candidate|Change|Validation macro|Selected round|Test during search|','|---|---|---:|---:|---|']
    for v in result['variants']:
        score='—' if v.get('validation_accuracy') is None else f"{v['validation_accuracy']*100:.2f}%"
        label=VARIANTS.get(v['id'],{}).get('label',v['label'])
        lines.append(f"|{v['id']}|{label}|{score}|{v.get('selected_round','—')}|Not evaluated|")
    lines+=['','## Three-Seed Results for the Locked Recipe','','|seed|Round-selection baseline A macro|Candidate macro|Change (percentage points)|Original fixed 50-round macro|','|---|---:|---:|---:|---:|']
    for p in result['pairs']:
        old=read_json(ROOT/'runs'/f"full-fedbrain-s{p['seed']}"/'result.json')['metrics']['site_mean_accuracy']
        a,b=p['baseline']['site_mean_accuracy'],p['optimized']['site_mean_accuracy']
        lines.append(f"|{p['seed']}|{a*100:.2f}%|{b*100:.2f}%|{(b-a)*100:+.2f}|{old*100:.2f}%|")
    lines+=['',f"- Original fixed 50-round macro mean: {comp['original_macro_mean']*100:.2f}%.",
            f"- Same validation round-selection baseline A: {comp['baseline_macro_mean']*100:.2f}%; optimized: {comp['optimized_macro_mean']*100:.2f}% ± {comp['optimized_macro_std']*100:.2f} percentage points.",
            f"- Optimized pooled subject accuracy mean: {comp['optimized_pooled_mean']*100:.2f}%.",
            f"- Paired mean change vs A: {comp['delta_macro']*100:+.2f} percentage points; {comp['success_count']}/3 seeds improved; passes preregistered >=2pp with at least 2/3 improved: {comp['pass_gate']}.",
            f"- Matched-mask/noise FedAvg with the same best-val round selection: {comp['fedavg_best_val_macro_mean']*100:.2f}% macro.", '',
            '## Rationale and Interpretation', '',
            '- Early stopping on validation addresses the problem of saving only the final round and discarding a better-validated model.',
            '- The batch convention (sumCE_batch+DA_batch)/actual batch adjusts the relative scale of classification and alignment; it is not a strict equivalence to the full-data Eq. 16 or a restoration of any source implementation.',
            '- After aggregation/broadcast, the train hidden means are recomputed with the current shared encoder, avoiding alignment to the coordinates of the previous round\'s local encoder.',
            '- L2 regularization 0.01 is an anti-overfitting extension; sparse 40→400 is a convergence diagnostic and is not claimed to be a specified hyperparameter.', '',
            '## Conclusion Boundaries', '',*['- '+n for n in result['notes']], '',
            'Whether the target macro accuracy is met must be read from real test values; validation accuracy is not a substitute for test results.', '']
    (ROOT/'docs'/'OPTIMIZATION_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['search','confirm','all'],default='all')
    args=p.parse_args()
    if args.phase in ('search','all'):
        search()
    if args.phase in ('confirm','all'):
        confirm()


if __name__=='__main__':
    main()
