# SPDX-License-Identifier: MIT
"""Verify real federation traces, parameter arithmetic, and non-invasive probes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from fedbrain.runner import atomic_json,read_json


def audit(run_id,parity_with=None):
    directory=ROOT/'runs'/run_id
    result=read_json(directory/'result.json')
    index=read_json(directory/'federation'/'index.json')
    assert result['status']=='completed' and index['status']=='completed'
    assert index['completed_rounds']==result['round']==result['config']['rounds']
    events=index['events']
    assert [e['sequence'] for e in events]==list(range(1,len(events)+1))
    assert all(a['monotonic_ns']<=b['monotonic_ns'] for a,b in zip(events,events[1:]))
    sites=index['site_order']
    checks=[]
    for round_number in range(1,index['completed_rounds']+1):
        frame=read_json(directory/'federation'/f'round-{round_number:04d}.json')
        assert frame['status']=='completed' and set(frame['clients'])==set(sites)
        subset=[e for e in events if e['round']==round_number]
        assert [e['site'] for e in subset if e['event']=='client_started']==sites
        assert [e['site'] for e in subset if e['event']=='client_uploaded']==sites
        aggregate=[e for e in subset if e['phase']=='aggregate']
        broadcast=[e for e in subset if e['event']=='broadcast_verified']
        pending=[e for e in subset if e['event']=='broadcast_pending']
        assert len(aggregate)==len(broadcast)==len(pending)==1
        assert aggregate[0]['sequence']<pending[0]['sequence']<broadcast[0]['sequence']
        assert frame['aggregation']['equal_mean_max_error']<=1e-6
        assert frame['after']['shared_consensus_max_error']==0.
        assert frame['after']['private_preserved']
        assert all(frame['after']['private_preserved_by_site'].values())
        uploaded=np.array([frame['clients'][s]['shared_sketch'] for s in sites])
        actual=np.array(frame['aggregation']['shared_sketch'])
        sketch_error=float(np.max(np.abs(uploaded.mean(axis=0)-actual)))
        assert sketch_error<=1e-6
        for site in sites:
            client=frame['clients'][site]
            assert client['update_l2']>0 and client['upload_bytes']==index['metadata']['shared_parameter_count']*4
            assert client['private_hash']==frame['after']['sites'][site]['private_hash']
            np.testing.assert_allclose(frame['before']['sites'][site]['shared_sketch'],frame['before']['shared_sketch'],atol=0,rtol=0)
            np.testing.assert_allclose(frame['after']['sites'][site]['shared_sketch'],actual,atol=0,rtol=0)
        checks.append({'round':round_number,'clients':len(sites),'all_local_updates_nonzero':True,
                       'aggregation_max_error':frame['aggregation']['equal_mean_max_error'],
                       'sketch_aggregation_max_error':sketch_error,'shared_consensus_max_error':0.,
                       'private_weights_preserved':True,
                       'server_update_l2':frame['aggregation']['server_delta_l2']})
    parity=None
    if parity_with:
        old=read_json(ROOT/'runs'/parity_with/'predictions.json')
        new=read_json(directory/'predictions.json')
        previous={p['id']:p for p in old}
        assert set(previous)==set(p['id'] for p in new)
        maximum=max(abs(p['asd_probability']-previous[p['id']]['asd_probability']) for p in new)
        assert maximum<=1e-6
        parity={'previous_run':parity_with,'same_test_subjects':True,
                'max_probability_error':maximum,'observability_is_noninvasive':True}
    evidence={'run_id':run_id,'passed':True,'rounds':checks,'events':len(events),
              'fixed_train_only_projection':index['metadata']['projection'],
              'shared_parameter_count':index['metadata']['shared_parameter_count'],
              'training_subjects':sum(s['train_subject_count'] for s in index['metadata']['sites'].values()),
              'parity':parity}
    atomic_json(ROOT/'artifacts'/f'{run_id}-audit.json',evidence)
    print({k:v for k,v in evidence.items() if k!='fixed_train_only_projection'})
    return evidence


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-id',default='federation-demo-s42')
    p.add_argument('--parity-with')
    args=p.parse_args()
    audit(args.run_id,args.parity_with)


if __name__=='__main__':
    main()
