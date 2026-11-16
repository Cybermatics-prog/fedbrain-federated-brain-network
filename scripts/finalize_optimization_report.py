"""Merge independently matched controls and explain the preregistered null result.
No model training, test re-evaluation, or candidate reselection occurs here.
"""
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from fedbrain.runner import atomic_json,read_json
from optimize import RESULT,report


def main():
    result=read_json(RESULT)
    if result.get('status')!='completed':
        raise RuntimeError('Wait for all locked confirmations')
    control=read_json(ROOT/'artifacts'/'optimized_fedavg_matched.json')
    assert control['selected_variant']==result['selected_variant']
    assert [r['seed'] for r in control['records']]==[42,43,44]
    checks=[]
    keys=('seed','rounds','learning_rate','local_epochs','batch_size','hidden_dim','noise_scale',
          'max_windows','max_features','sparse_max_iter','sparse_windows_per_subject','lambda_aug',
          'coupling','clip_norm','fold','test_size','validation_fraction','weight_decay','checkpoint_selection')
    for pair,base in zip(result['pairs'],control['records']):
        assert pair['seed']==base['seed']
        target=ROOT/'runs'/pair['optimized_id']
        source=ROOT/'runs'/base['id']
        a,b=read_json(target/'result.json'),read_json(source/'result.json')
        for key in keys:
            assert a['config'][key]==b['config'][key],(pair['seed'],key)
        assert read_json(target/'split.json')==read_json(source/'split.json')
        pa,pb=np.load(target/'preprocessing.npz'),np.load(source/'preprocessing.npz')
        assert np.array_equal(pa['indices'],pb['indices'])
        np.testing.assert_allclose(pa['mean'],pb['mean'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(pa['std'],pb['std'],rtol=0,atol=1e-12)
        checks.append({'seed':pair['seed'],'identical_mask':True,'identical_train_scaling':True,
                       'identical_subject_split':True,'n_features':len(pa['indices'])})
    comp=result['comparison']
    if not result.get('matched_control_audited'):
        result['fedavg_original_mask']=result['fedavg']
        comp['fedavg_original_mask_best_val_macro_mean']=comp['fedavg_best_val_macro_mean']
    result['fedavg']=control['records']
    comp['fedavg_best_val_macro_mean']=float(np.mean([r['metrics']['site_mean_accuracy'] for r in control['records']]))
    comp['fedavg_matched_mask']=True
    comp['delta_vs_matched_fedavg_macro']=comp['optimized_macro_mean']-comp['fedavg_best_val_macro_mean']
    original=[read_json(ROOT/'runs'/f'full-fedbrain-s{s}'/'result.json')['metrics'] for s in (42,43,44)]
    comp['original_pooled_mean']=float(np.mean([m['accuracy'] for m in original]))
    comp['delta_original_macro']=comp['optimized_macro_mean']-comp['original_macro_mean']
    comp['delta_original_pooled']=comp['optimized_pooled_mean']-comp['original_pooled_mean']
    result['matched_control_audited']=True
    result['matched_control_checks']=checks
    result['outcome']='passed_minimum_effect' if comp['pass_gate'] else 'no_stable_improvement'
    result['deployment']='validation_locked_candidate' if comp['pass_gate'] else 'keep_original_default'
    result['interpretation']='The validation gain did not transfer stably to the test set; a small pooled increase is not a macro improvement under the paper\'s protocol.'
    extra='E uses the 400-step mask; FedAvg is additionally matched with the same 400-step mask/standardization/split/round selection, and the masks are identical across all three seeds.'
    if extra not in result['notes']:
        result['notes'].append(extra)
    atomic_json(RESULT,result)
    report(result)
    path=ROOT/'docs'/'OPTIMIZATION_RESULTS.md'
    text=path.read_text(encoding='utf-8')
    table=['','## The Original Baseline Must Not Be Hidden','','|Mean over the same three seeds|Site macro|Pooled subjects|','|---|---:|---:|',
           f"|Original fixed 50 rounds|{comp['original_macro_mean']*100:.2f}%|{comp['original_pooled_mean']*100:.2f}%|",
           f"|A: validation round selection only|{comp['baseline_macro_mean']*100:.2f}%|{comp['baseline_pooled_mean']*100:.2f}%|",
           f"|E: validation-selected candidate|{comp['optimized_macro_mean']*100:.2f}%|{comp['optimized_pooled_mean']*100:.2f}%|",'',
           f"Candidate vs original macro change **{comp['delta_original_macro']*100:+.2f} percentage points**; pooled change {comp['delta_original_pooled']*100:+.2f} percentage points — do not report only the more flattering metric.",'',
           '## Failure Autopsy and Stop Decision','',
           f"1. The failure is in generalization: seed42 validation +{result['validation_gain_over_A']*100:.2f}pp, but the three-seed test average vs A is only +{comp['delta_macro']*100:.2f}pp, below the 2pp gate set before the run; seed44 is negative, and the original macro is higher too.",
           '2. Training being unlearnable and obvious pipeline errors have been ruled out: the existing real train-only 8-subject power gate passes; full private/gated restoration, loss gradients and the current broadcast-representation timing all have unit tests; an independent audit separately checked original predictions/subject exclusivity/checkpoint replay.',
           '3. Remaining mechanism: 400 steps still do not solve the joint nonsmooth optimization of Eq11, and 77 validators with the smallest site at 6 subjects make round/candidate selection visibly unstable; a single KKI validation classification flip changes macro by about 2.78pp.','',
           'This round of bounded five candidates is complete; B/C/D test sets will not be opened, and D will not be reselected based on E\'s test failure. Keep the original interface default model and do not declare the accuracy optimization successful.','',
           '## Next-Round Priorities (Not Executed; Not Claimed Effective)','',
           '1. First solve Eq11 with a joint proximal/primal-dual/ADMM method; it must pass analytic counterexamples and KKT/dual residuals before measuring generalization.',
           '2. Select lambda_aug and the feature count in nested validation instead of fixing top512 with unconverged coefficients; the outer fixed repeated cross-validation is used only for final comparison.',
           '3. If the first two items still give no gain, separately preregister multi-scale dynamic FC + static FC fusion; do not first enlarge the network or the noise.','',
           '## Matched-Control Audit','',
           f"400-step same-mask/noise/round-selection FedAvg mean {comp['fedavg_best_val_macro_mean']*100:.2f}% macro; the candidate is {comp['delta_vs_matched_fedavg_macro']*100:+.2f}pp relative to it, which does not support superiority over FedAvg.",'',
           '|seed|Identical indices|Identical training scaler|Identical subject split|','|---|---|---|---|']
    for check in checks:
        table.append(f"|{check['seed']}|Yes|Yes|Yes|")
    table+=['',f"The original 40-step mask + best-val FedAvg is {comp['fedavg_original_mask_best_val_macro_mean']*100:.2f}% macro, stored separately as an auxiliary control with different preprocessing; it is not claimed to be the same mask.",'']
    path.write_text(text+'\n'.join(table),encoding='utf-8')
    atomic_json(ROOT/'artifacts'/'optimization_canonical_numbers.json',
                {'source':'optimization_results.json, locked paired result.json, matched control result.json',
                 'selection_locked_at':result['selection_locked_at'],'comparison':comp,
                 'matched_control_checks':checks,'outcome':result['outcome']})
    print('No training or retesting. Canonical result:',comp)


if __name__=='__main__':
    main()
