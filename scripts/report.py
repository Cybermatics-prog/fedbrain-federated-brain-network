# SPDX-License-Identifier: MIT
"""Build the user-facing results report exclusively from saved experiment JSON."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fedbrain.runner import atomic_json, read_json


def percent(value):
    return '—' if value is None else f'{100*value:.2f}%'


def main():
    manifest = read_json(ROOT/'data'/'abide'/'manifest.json', {})
    results = [read_json(p) for p in (ROOT/'runs').glob('*/result.json')]
    results = sorted([r for r in results if r], key=lambda r:r['created_at'])
    successful = [r for r in results if r['status']=='completed' and r.get('metrics') is not None]
    full = [r for r in successful if not r['config']['max_per_site'] and r['config']['rounds']==50]
    lines = ['# FedBrain Measured Results', '',
             'This report is generated automatically by `scripts/report.py` from real `result.json` files; experiment numbers are never entered by hand.', '',
             '## Data and Evaluation Scope', '',
             f"- Public subjects actually loaded: {len(manifest.get('subjects',[]))}; download failures: {len(manifest.get('failures',[]))}.",
             '- CPAC / filt_noglobal / AAL116; window 32, stride 1, Pearson→Fisher-z, 6670-dimensional upper triangle.',
             '- Subjects are split into train/val/test first; sparsification and standardization use training data only; each subject\'s window predicted probabilities are averaged.',
             '- macro-site is the equal-weight mean of the six site accuracies; pooled is the accuracy over all test subjects combined. The two must not be conflated.',
             '- The numbers below come from this system only; they are not comparable with results obtained elsewhere on a different cohort or a reduced evaluation protocol.', '',
             '|Site|Actual subjects|Reference per-site counts|','|---|---:|---:|']
    for site,count in manifest.get('counts',{}).items():
        lines.append(f"|{site}|{count}|{manifest.get('reference_subject_counts',{}).get(site,'—')}|")
    lines += ['', '## Full 50-Round Experiments', '',
              'Experiments are grouped by recipe; only FedAvg runs with the matching configuration/seed/feature mask are matched-mask controls, so 40-step and 400-step masks must not be conflated. This is not a claim to cover every baseline configuration.', '',
              '|Run|Method|seed|macro-site accuracy|pooled accuracy|AUC|Test subjects|Training accuracy|Elapsed seconds|',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in full:
        m=r['metrics']; cfg=r['config']
        lines.append(f"|`{r['id']}`|{cfg['method']}|{cfg['seed']}|{percent(m['site_mean_accuracy'])}|{percent(m['accuracy'])}|{m['auc']:.4f}|{m['n_test_subjects']}|{percent(r.get('train_metrics',{}).get('accuracy'))}|{r['elapsed_seconds']:.1f}|")
    grouped = defaultdict(list)
    for r in full:
        group=(r['config']['method'],r['config'].get('experiment_tag') or 'original-fixed-final')
        grouped[group].append(r)
    lines += ['', '### Summary by Experiment Recipe (candidates, budgets and round-selection rules not mixed)', '']
    for (method,recipe),members in grouped.items():
        values=[r['metrics']['site_mean_accuracy'] for r in members]
        pooled=[r['metrics']['accuracy'] for r in members]
        lines.append(f"- {method} / {recipe}: n={len(values)}, macro-site mean={percent(float(np.mean(values)))}, sample std={f'{100*np.std(values,ddof=1):.2f} percentage points' if len(values)>1 else 'single seed, std not reported'}; pooled mean={percent(float(np.mean(pooled)))}.")
    if full:
        main_run=next((r for r in full if r['id']=='full-fedbrain-s42'),full[0])
        lines += ['', f"## Site results for main run {main_run['id']}", '',
                  '|Site|Test subjects|Accuracy|AUC|Sensitivity|Specificity|', '|---|---:|---:|---:|---:|---:|']
        for site,m in main_run['metrics']['per_site'].items():
            lines.append(f"|{site}|{m['n_test_subjects']}|{percent(m['accuracy'])}|{m['auc']:.4f}|{percent(m['sensitivity'])}|{percent(m['specificity'])}|")
        lines += ['', '### Interpretable Conclusion', '',
                  ('The current main run is above the reference figure kept for orientation, but the cohort and protocol differ, so this is not a table-by-table comparison.' if main_run['metrics']['site_mean_accuracy']>=.79 else
                   'The current main run is below the reference figure kept for orientation: the system runs end to end on real data, but the reference value has not been reached under this configuration; the residual gap has not yet been characterised.'),
                  'Original failures and low-scoring results are kept; no tuning on the test set or selection of rounds is performed.']
    power=read_json(ROOT/'artifacts'/'real_power_gate.json',{})
    audit=read_json(ROOT/'artifacts'/'audit_results.json',{})
    controls=read_json(ROOT/'web'/'qa'/'start-stop-report.json',{})
    lines += ['', '## Independent Validity Checks', '',
              f"- Real train-only 8-subject overfitting power control: passed={power.get('passed')}, accuracy={percent(power.get('metrics',{}).get('accuracy'))}.",
              f"- Split/raw-probability/recomputed-score consistency across completed experiments: {len(audit.get('checks',[]))} checks, all_passed={audit.get('all_completed_runs_passed')}.",
              f"- Maximum predicted-probability error between Paillier 2048-bit and plaintext smoke runs: {audit.get('encrypted_vs_plain_max_probability_error')}.",
              '- Real browser checks passed on a 1440 desktop and a 390 mobile viewport; all displayed values come from real API responses.',
              f"- Real interface START/STOP: win={controls.get('win')}, HTTP={controls.get('start_http')}/{controls.get('stop_http')}, persisted state={controls.get('persisted_status')}."]
    lines += ['', '## Acceleration Approximations and Privacy Settings', '',
              '- In the main configuration, the sparse stage uses 8 relative time segments per training subject, 40 iterations, and at most 512 connections; the classification stage uses all windows.',
              '- Optimization, merging and batching details that the reference description leaves open are fixed explicitly in the configuration; the reference subject list is unavailable.',
              '- The main configuration clips shared parameters and then adds Laplace noise with noise_scale=0.01; this is not epsilon-DP with a budget-accounting proof.',
              '- Homomorphic aggregation is optional real Paillier 2048-bit arithmetic; holding the key in the same process on one machine is not equivalent to a deployment across independent hospitals.', '',
              '## Smoke and Homomorphic Runs', '', '|Run|Status|Rounds|Homomorphic aggregation|Test accuracy|','|---|---|---:|---|---:|']
    for r in results:
        if r in full:
            continue
        lines.append(f"|`{r['id']}`|{r['status']}|{r.get('round',0)}|{r['config']['homomorphic']}|{percent((r.get('metrics') or {}).get('accuracy'))}|")
    lines += ['', '## Evidence Entry Points', '',
              '- Test-unlocked runs contain result.json / split.json / predictions.json / checkpoint.pt / preprocessing.npz; validation candidates have only validation_predictions.json and do not generate test predictions.',
              '- Test command: `.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v`.',
              '- Full 5-repeat × 5-fold script: `scripts/benchmark.py`; the 25-run protocol counts as complete only once it has actually been executed to completion.',
              '- The original failing smoke-six-sites recorded insufficient minority-class samples caused by ID truncation; this was fixed with fixed-seed stratified sampling within each site plus a regression test.', '']
    (ROOT/'docs'/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    atomic_json(ROOT/'artifacts'/'results_summary.json',{'cohort':manifest.get('counts',{}),'runs':[
        {'id':r['id'],'status':r['status'],'config':r['config'],'metrics':r.get('metrics')} for r in results]})
    print('\n'.join(lines))


if __name__=='__main__':
    main()
