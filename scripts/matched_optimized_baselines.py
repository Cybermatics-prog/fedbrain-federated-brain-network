# SPDX-License-Identifier: MIT
"""Match the already validation-locked preprocessing/regularization for FedAvg.
Writes separate evidence only; never changes the chosen FedBrain recipe.
"""
import sys
from dataclasses import asdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from fedbrain.runner import RunConfig,atomic_json,read_json
from optimize import SELECTION,configuration,run_or_reuse


def main():
    selection=read_json(SELECTION)
    if not selection or not selection['test_was_not_evaluated_for_selection']:
        raise RuntimeError('Require previously persisted validation-only selection')
    variant=selection['selected_variant']
    records=[]
    for seed in (42,43,44):
        cfg=configuration(variant,seed,False,method='fedavg')
        # Fresh deterministic same-budget mask fit avoids depending on future
        # FedBrain runs. Mask identity is independently checked after both finish.
        cfg.update(feature_run_id=None,experiment_tag='optimized-matched-FedAvg')
        run_id=f'opt-matched-FedAvg-s{seed}'
        result=run_or_reuse(run_id,cfg)
        requested=asdict(RunConfig(**cfg))
        if requested!=result['config']:
            raise RuntimeError('Completed comparator configuration is stale')
        records.append({'id':run_id,'seed':seed,'metrics':result['metrics'],
                        'selected_round':result['selected_round'],'source_hashes':result['source_hashes']})
        atomic_json(ROOT/'artifacts'/'optimized_fedavg_matched.json',
                    {'selected_variant':variant,'selection_locked_at':selection['locked_at'],
                     'records':records,'purpose':'fair matched preprocessing/noise/regularization/selection, never candidate selection'})
        print('Matched locked preprocessing FedAvg:',seed,result['metrics']['site_mean_accuracy'],flush=True)


if __name__=='__main__':
    main()
