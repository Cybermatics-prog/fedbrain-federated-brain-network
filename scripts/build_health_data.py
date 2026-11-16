"""Build/audit real health-data views without loading or training any model."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from fedbrain.data import load_subjects


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--describe-only',action='store_true')
    args=parser.parse_args()
    subjects=load_subjects(ROOT/'data'/'abide')
    counts=Counter(s['label'] for s in subjects)
    ages=[s['age'] for s in subjects if s.get('age') is not None]
    print(json.dumps({'total_subjects':len(subjects),'HC':counts[0],'ASD':counts[1],
                      'age_min':min(ages),'age_max':max(ages),'age_missing':len(subjects)-len(ages)},ensure_ascii=False),flush=True)
    if args.describe_only:
        return
    from fedbrain.health_data import HealthDataStore
    import numpy as np
    started=time.perf_counter()
    store=HealthDataStore(ROOT/'data'/'abide',ROOT/'artifacts'/'health_data')
    overview=store.overview()
    assert overview['total_subjects']==len(subjects)
    assert overview['n_hc']==counts[0] and overview['n_asd']==counts[1]
    assert overview['selected_subjects']==len(subjects)
    assert sum(s['total'] for s in overview['site_counts'])==len(subjects)
    assert sum(overview['age_histogram']['hc'])+overview['age_histogram']['missing_hc']==counts[0]
    assert sum(overview['age_histogram']['asd'])+overview['age_histogram']['missing_asd']==counts[1]
    for group in ('HC','ASD'):
        fc=np.asarray(overview['groups'][group]['mean_fc'])
        assert fc.shape==(116,116) and np.isfinite(fc).all()
        np.testing.assert_allclose(fc,fc.T,atol=1e-7)
        assert np.max(np.abs(fc))<=1.+1e-7
    np.testing.assert_allclose(overview['difference']['fc'],np.array(overview['groups']['ASD']['mean_fc'])-np.array(overview['groups']['HC']['mean_fc']),atol=1e-6)
    hc=next(s for s in overview['subjects'] if s['label']==0)
    detail=store.subject(hc['id'])
    assert detail['label']==0 and detail['reference']['hc_count']==counts[0]-1
    assert len(detail['signals']['series'])==3
    assert all(len(s['values'])==detail['n_frames'] for s in detail['signals']['series'])
    assert len(detail['signals']['frames'])==detail['n_frames']
    site=store.overview(site='KKI')
    assert site['selected_subjects']==33 and site['n_hc']==21 and site['n_asd']==12
    empty=store.overview(age_min=110,age_max=120)
    assert empty['selected_subjects']==0
    assert empty['groups']['HC']['mean_fc'] is None
    evidence={'passed':True,'seconds':time.perf_counter()-started,
              'total_subjects':overview['total_subjects'],'hc':overview['n_hc'],'asd':overview['n_asd'],
              'site_counts':overview['site_counts'],'age_summary':overview['age_summary'],
              'matrix_size':[116,116],'hc_self_excluded_reference':detail['reference']['hc_count'],
              'checked_healthy_subject':hc['id'],'bold_frames':detail['n_frames'],
              'bold_series':len(detail['signals']['series']),
              'kki_filter_verified':True,'empty_group_verified':True,'model_checkpoint_required':False}
    (ROOT/'artifacts'/'health_data_overview.json').write_text(json.dumps(overview,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    (ROOT/'artifacts'/'health_data_audit.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(evidence,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
