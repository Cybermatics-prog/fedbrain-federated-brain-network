# SPDX-License-Identifier: MIT
"""Preregistered easiest REAL-data training power gate, never a test result."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from fedbrain.data import load_subjects
from fedbrain.model import FedBrainModel
from fedbrain.runner import RunConfig, _predict_subject, _subject_features, atomic_json, read_json
from fedbrain.metrics import grouped_metrics


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    directory=ROOT/'runs'/'full-fedbrain-s42'
    checkpoint=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=False)
    config=RunConfig(**checkpoint['config'])
    config.max_windows=32
    split=read_json(directory/'split.json')
    ids=set(s['id'] for s in split['train'])
    subjects=[s for s in load_subjects(config.data_dir) if s['id'] in ids and s['site']=='NYU']
    tiny=[]
    for label in (0,1):
        tiny.extend(sorted([s for s in subjects if s['label']==label],key=lambda s:s['id'])[:4])
    assert len(tiny)==8 and all(s['id'] in ids for s in tiny)
    ix=checkpoint['indices']
    arrays={s['id']:np.asarray((_subject_features(s,config)[:,ix]-checkpoint['mean'][ix])/checkpoint['std'][ix],dtype=np.float32) for s in tiny}
    x=torch.from_numpy(np.concatenate([arrays[s['id']] for s in tiny]))
    y=torch.from_numpy(np.concatenate([np.full(len(arrays[s['id']]),s['label'],dtype=np.float32) for s in tiny]))
    model=FedBrainModel(len(ix),8)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    loss_history=[]
    for step in range(300):
        model.train()
        out=model(x)
        loss=F.binary_cross_entropy_with_logits(out['mixed_logit'],y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (0,49,99,199,299):
            loss_history.append({'step':step+1,'loss':float(loss.detach())})
    predictions=[{'id':s['id'],'site':s['site'],'label':s['label'],**_predict_subject(model,arrays[s['id']],'cpu')} for s in tiny]
    metrics=grouped_metrics(predictions)
    result={'purpose':'REAL-data training power gate only; NOT held-out performance',
            'source_run':'full-fedbrain-s42','all_subjects_from_train_only':True,
            'ids':[s['id'] for s in tiny],'n_subjects':8,'max_windows_per_subject':32,
            'optimizer':'Adam','learning_rate':.01,'steps':300,'hidden_dim':8,
            'noise_scale':0,'alignment_weight':0,'win_accuracy':.95,
            'loss_history':loss_history,'metrics':metrics,'passed':metrics['accuracy']>=.95}
    atomic_json(ROOT/'artifacts'/'real_power_gate.json',result)
    print(result)
    if not result['passed']:
        raise SystemExit('REAL power gate FAILED; no scientific conclusion allowed')


if __name__=='__main__':
    main()
