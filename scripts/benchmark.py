"""Sequential fixed-budget repeated subject-stratified cross-validation.

Never picks the best held-out fold/round. Every seed/fold/method is reported.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fedbrain.runner import RunConfig, atomic_json, read_json, run_training


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=[42,43,44,45,46])
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--methods', nargs='+', default=['fedbrain','fedavg'])
    p.add_argument('--config', type=Path, default=ROOT/'configs'/'fedbrain.json')
    p.add_argument('--tag', default='repeated-cv')
    args = p.parse_args()
    base = read_json(args.config)
    records = []
    summary_path = ROOT / 'runs' / (args.tag + '-summary.json')
    for seed in args.seeds:
        for fold in range(args.folds):
            feature_id = None
            for method in args.methods:
                run_id = f'{args.tag}-s{seed}-f{fold}-{method}'
                cfg = {**base, 'seed':seed, 'fold':fold, 'n_splits':args.folds, 'method':method,
                       'feature_run_id':feature_id}
                result = read_json(ROOT / 'runs' / run_id / 'result.json')
                if not result or result.get('status') != 'completed':
                    result = run_training(RunConfig(**cfg), run_id)
                if method == args.methods[0]:
                    feature_id = run_id
                records.append({'id':run_id, 'seed':seed, 'fold':fold, 'method':method,
                                'metrics':result['metrics'], 'config':cfg})
                summaries = {}
                for m in args.methods:
                    values = [r['metrics']['site_mean_accuracy'] for r in records if r['method']==m]
                    if values:
                        summaries[m] = {'completed_runs':len(values), 'macro_site_accuracy_mean':float(np.mean(values)),
                                        'std_across_folds_and_seeds':float(np.std(values, ddof=1)) if len(values)>1 else None,
                                        'all_scores':values}
                atomic_json(summary_path, {'records':records, 'summary':summaries,
                                          'target':{'seeds':args.seeds, 'folds':args.folds, 'methods':args.methods},
                                          'complete':len(records)==len(args.seeds)*args.folds*len(args.methods)})
    print(json.dumps(read_json(summary_path)['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
