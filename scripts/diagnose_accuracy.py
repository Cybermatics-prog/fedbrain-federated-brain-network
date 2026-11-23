# SPDX-License-Identifier: MIT
"""Read existing predictions only: diagnose demo undertraining vs main generalization."""
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parent.parent

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def describe(run_id):
    directory=ROOT/'runs'/run_id
    result=read(directory/'result.json')
    predictions=read(directory/'predictions.json')
    n=len(predictions)
    y=[p['label'] for p in predictions]
    correct=sum(p['predicted_label']==p['label'] for p in predictions)
    assert abs(correct/n-result['metrics']['accuracy'])<1e-12
    positive=sum(y)
    called_positive=sum(p['predicted_label'] for p in predictions)
    tps=sum(p['label']==1 and p['predicted_label']==1 for p in predictions)
    row={'run_id':run_id,'rounds':result['config']['rounds'],'experiment_tag':result['config'].get('experiment_tag'),
         'train_pooled':result['train_metrics']['accuracy'],'train_macro':result['train_metrics']['site_mean_accuracy'],
         'validation_pooled':result['validation_metrics']['accuracy'],
         'test_pooled':correct/n,'test_macro':result['metrics']['site_mean_accuracy'],
         'test_auc':result['metrics']['auc'],'n_test':n,'n_asd':positive,'n_hc':n-positive,
         'predicted_asd':called_positive,'predicted_hc':n-called_positive,
         'correct_asd':tps,'sensitivity':tps/positive,'specificity':result['metrics']['specificity'],
         'majority_class_accuracy':max(positive,n-positive)/n,
         'score_means':{str(label):mean(p['asd_probability'] for p in predictions if p['label']==label) for label in (0,1)},
         'expert_readout_diagnostic_not_retrained_ablation':{
             expert:sum((p[key]>=.5)==p['label'] for p in predictions)/n
             for expert,key in [('global','global_probability'),('private','private_probability')]}}
    return row,result

def main():
    ids=['federation-demo-s42']+[f'full-fedbrain-s{s}' for s in (42,43,44)]+[f'full-fedavg-s{s}' for s in (42,43,44)]
    rows=[]
    results={}
    for run_id in ids:
        row,result=describe(run_id)
        rows.append(row)
        results[run_id]=result
    demo=results[ids[0]]
    original=results['full-fedbrain-s42']
    a=read(ROOT/'runs'/ids[0]/'split.json')
    b=read(ROOT/'runs'/'full-fedbrain-s42'/'split.json')
    same_split=all({s['id'] for s in a[k]}=={s['id'] for s in b[k]} for k in ('train','val','test'))
    keys=('seed','learning_rate','local_epochs','batch_size','hidden_dim','noise_scale','alignment_weight',
          'max_per_site','max_windows','max_features','clip_norm','test_size','validation_fraction')
    same_config=all(demo['config'][k]==original['config'][k] for k in keys)
    evidence={'generated_from_existing_records_only':True,'no_training_or_threshold_selection':True,
              'runs':rows,'demo_vs_original42':{'same_subject_split':same_split,'same_learning_configuration_except_rounds':same_config,
                  'max_validation_accuracy_prefix_difference':max(abs(x['val_accuracy']-y['val_accuracy']) for x,y in zip(demo['history'],original['history'])),
                  'test_pooled_delta_8_to_50':original['metrics']['accuracy']-demo['metrics']['accuracy']},
              'three_seed_original':{'train_pooled_mean':mean(r['train_pooled'] for r in rows[1:4]),
                  'test_pooled_mean':mean(r['test_pooled'] for r in rows[1:4]),
                  'test_macro_mean':mean(r['test_macro'] for r in rows[1:4]),
                  'fedavg_test_macro_mean':mean(r['test_macro'] for r in rows[4:7])}}
    path=ROOT/'artifacts'/'accuracy_diagnosis.json'
    path.write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(evidence,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
