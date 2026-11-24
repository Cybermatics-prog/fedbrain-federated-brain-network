# SPDX-License-Identifier: MIT
"""Local research API: source-derived health data and recorded protocol facts."""
from __future__ import annotations

import json
import re
import threading
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from .data import REFERENCE_COUNTS, SITE_ORDER, load_subjects
from .runner import ROOT, RunConfig, atomic_json, infer_subject, read_json, run_training

app = FastAPI(title='FedBrain Research Studio', version='0.1.0', docs_url='/api/docs')
DATA_DIR = ROOT / 'data' / 'abide'
RUN_DIR = ROOT / 'runs'
_training_lock = threading.Lock()
_health_lock = threading.Lock()
_health_store = None
_active = {'id': None, 'thread': None, 'stop': None}
_DEVICE = {'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
           'cuda': torch.cuda.is_available(), 'torch': torch.__version__,
           'memory_gb': round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if torch.cuda.is_available() else None}


def _safe_run(run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,120}', run_id):
        raise HTTPException(status_code=400, detail='Invalid run identifier')
    path = RUN_DIR / run_id
    if not path.is_dir():
        raise HTTPException(status_code=404, detail='Run not found')
    return path


def _runs():
    runs = []
    if RUN_DIR.is_dir():
        for path in RUN_DIR.iterdir():
            if not path.is_dir():
                continue
            result = read_json(path / 'result.json')
            if result:
                # A previous process may have terminated before final status write.
                if result['status'] in ('running', 'preparing') and _active['id'] != result['id']:
                    try:
                        import psutil
                        alive = psutil.pid_exists(result.get('process_id', -1))
                    except ImportError:
                        alive = True
                    if not alive:
                        result = {**result, 'status': 'interrupted', 'error': 'Training process is no longer active'}
                    else:
                        result = {**result, 'external_run': True}
                result = {**result, 'federation_available': (path / 'federation' / 'index.json').is_file()}
                runs.append(result)
    return sorted(runs, key=lambda r: r.get('created_at', ''), reverse=True)


def _dataset():
    manifest = read_json(DATA_DIR / 'manifest.json', {})
    subjects = manifest.get('subjects', [])
    return {'status': manifest.get('status', 'not_downloaded'), 'subjects': len(subjects),
            'total_subjects': len(subjects),
            'candidate_subjects': len(manifest.get('candidates', [])),
            'sites': [{'id': site, 'name': site, 'count': sum(s['site'] == site for s in subjects),
                       'asd': sum(s['site'] == site and s['label'] == 1 for s in subjects),
                       'hc': sum(s['site'] == site and s['label'] == 0 for s in subjects),
                       'reference_count': REFERENCE_COUNTS[site]} for site in SITE_ORDER],
            'pipeline': 'CPAC / filt_noglobal / AAL116', 'failures': len(manifest.get('failures', [])),
            'cohort_note': manifest.get('cohort_note', 'Public data download in preparation')}


@app.get('/api/health')
def health():
    return {'ok': True, 'version': '0.1.0'}


@app.get('/api/status')
def status():
    runs = _runs()
    summaries = [{k: v for k, v in r.items() if k not in ('history', 'traceback')} for r in runs]
    return {'dataset': _dataset(), 'device': _DEVICE,
            'active_run': next((r['id'] for r in runs if r['status'] in ('preparing', 'running')), None),
            'runs': summaries,
            'protocol': {'window': 32, 'stride': 1, 'atlas': 'AAL116', 'features': 6670,
                         'hidden_dim': 8, 'rounds': 50, 'split': 'subject-level stratified per site; held-out validation and test',
                         'reference_evaluation': '5 repeats × 5 folds (25 runs), not inferred from a single hold-out',
                         'noise_discrepancy': 'Earlier descriptions of the noise scale disagree (lambda_M=0.01 in the method description, 0.1 in the summary); this implementation defaults to 0.01 and records the actual configuration.',
                         'privacy_note': 'Parameter noise is not a proven epsilon-DP guarantee; Paillier is an optional real homomorphic aggregation, disabled by default.',
                         'implementation_note': 'Implementation choices such as sparse-optimization sampling and feature caps are disclosed in every run configuration.',
                         'clinical_note': 'Research and teaching demonstration only; not for clinical diagnosis.'}}


def _get_health_store():
    global _health_store
    if _health_store is None:
        with _health_lock:
            if _health_store is None:
                from .health_data import HealthDataStore
                _health_store = HealthDataStore(DATA_DIR, ROOT / 'artifacts' / 'health_data')
    return _health_store


@app.get('/api/health-data')
def health_data(site: str | None = None,
                age_min: float | None = Query(default=None, ge=0, le=120),
                age_max: float | None = Query(default=None, ge=0, le=120)):
    if age_min is not None and age_max is not None and age_min > age_max:
        raise HTTPException(status_code=422, detail='Minimum age cannot exceed maximum age')
    try:
        return _get_health_store().overview(site=site, age_min=age_min, age_max=age_max)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=f'Real health data are not ready yet: {exc}') from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get('/api/health-data/sites')
def health_sites():
    try:
        return _get_health_store().site_previews()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=f'Health data are not ready yet: {exc}') from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get('/api/health-data/subjects/{subject_id}')
def health_subject(subject_id: str):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,120}', subject_id):
        raise HTTPException(status_code=400, detail='Invalid subject identifier')
    try:
        return _get_health_store().subject(subject_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail='That subject is not in the public dataset') from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=f'Real health data are not ready yet: {exc}') from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get('/api/data-protection')
def data_protection(run_id: str | None = None):
    from .protection import describe_protection
    candidates = [item for item in _runs() if item.get('status') == 'completed'
                  and (item.get('federation_available') or item.get('config', {}).get('homomorphic'))]
    preferred = ('federation-demo-s42', 'encrypted-smoke-2048')
    candidates.sort(key=lambda item: (preferred.index(item['id']) if item['id'] in preferred else len(preferred)))
    selected = _safe_run(run_id) if run_id else (RUN_DIR / candidates[0]['id'] if candidates else None)
    profile = describe_protection(selected)
    profile['available_runs'] = [
        {'id': item['id'], 'status': item['status'], 'rounds_completed': item.get('round', 0),
         'homomorphic': bool(item.get('config', {}).get('homomorphic', False)),
         'noise_scale': item.get('config', {}).get('noise_scale')}
        for item in candidates]
    return profile


@app.get('/api/optimization')
def optimization():
    return read_json(ROOT / 'artifacts' / 'optimization_results.json',
                     {'status': 'pending', 'selected_variant': None, 'selected_run_id': None,
                      'variants': [], 'notes': ['Validation-based selection has not started; no simulated gains are shown.']})


@app.get('/api/runs')
def runs():
    return {'runs': [{k: v for k, v in r.items() if k not in ('history', 'traceback')} for r in _runs()]}


@app.get('/api/runs/{run_id}')
def run(run_id: str):
    path = _safe_run(run_id)
    result = read_json(path / 'result.json')
    if not result:
        raise HTTPException(status_code=404, detail='Run result not ready')
    return result


@app.get('/api/runs/{run_id}/federation')
def federation_index(run_id: str):
    path = _safe_run(run_id)
    recorded = read_json(path / 'federation' / 'index.json')
    if recorded:
        return recorded
    result = read_json(path / 'result.json', {})
    return {'available': False, 'schema_version': 1, 'run_id': run_id,
            'status': result.get('status'), 'site_order': SITE_ORDER,
            'events': [], 'rounds': [], 'completed_rounds': 0,
            'reason': 'legacy_run_no_recorded_federation_trace',
            'message': 'This historical run did not record per-client state, so a federation replay cannot be reconstructed. Choose an experiment that recorded its process.'}


@app.get('/api/runs/{run_id}/federation/{round_number}')
def federation_round(run_id: str, round_number: int):
    path = _safe_run(run_id)
    if not 1 <= round_number <= 500:
        raise HTTPException(status_code=422, detail='Round must be1..500')
    result = read_json(path / 'federation' / f'round-{round_number:04d}.json')
    if result is None:
        raise HTTPException(status_code=404, detail='That round was not recorded; no synthetic parameters or animation data are produced')
    return result


def _selected_run(run_id=None, completed=False):
    if run_id:
        path = _safe_run(run_id)
        if completed and not (path / 'checkpoint.pt').is_file():
            raise HTTPException(status_code=409, detail='This experiment has no completed model; choose a completed run')
        info = read_json(path / 'result.json', {})
        if completed and info.get('config', {}).get('defer_test') and not info.get('test_evaluated'):
            raise HTTPException(status_code=409, detail='This experiment was used for validation-only selection; test and individual inference remain locked')
        return path
    for item in _runs():
        path = RUN_DIR / item['id']
        eligible = not item.get('config', {}).get('defer_test') or item.get('test_evaluated')
        if not completed or ((path / 'checkpoint.pt').is_file() and eligible):
            return path
    if completed:
        raise HTTPException(status_code=409, detail='No trained model yet; complete one real-data training run first')
    return None


@app.get('/api/subjects')
def subjects(run_id: str | None = None):
    try:
        data = load_subjects(DATA_DIR)
    except FileNotFoundError:
        return {'subjects': []}
    run_dir = _selected_run(run_id)
    split = read_json(run_dir / 'split.json', {}) if run_dir else {}
    lookup = {str(s['id']): key for key, values in split.items() for s in values}
    result = [{**{k: s.get(k) for k in ('id', 'site', 'label', 'age')},
               'label_name': 'ASD' if s['label'] else 'HC', 'split': lookup.get(str(s['id']), 'unused')}
              for s in data]
    order = {'test': 0, 'val': 1, 'train': 2, 'unused': 3}
    result.sort(key=lambda s: (order.get(s['split'], 4), s['site'], s['id']))
    return {'subjects': result}


@app.get('/api/subjects/{subject_id}')
def subject(subject_id: str, run_id: str | None = None):
    run_dir = _selected_run(run_id, completed=True)
    try:
        return infer_subject(run_dir, subject_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class TrainingRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    rounds: int = 50
    seed: int = 42
    method: str = 'fedbrain'
    max_per_site: int = 0
    noise_scale: float = .01
    alignment_weight: float = .1
    max_features: int = 512
    max_windows: int = 0
    local_epochs: int = 1
    learning_rate: float = .001
    homomorphic: bool = False
    feature_run_id: str | None = None
    fold: int | None = None
    checkpoint_selection: str = 'final'
    alignment_normalization: str = 'legacy'
    alignment_reference: str = 'legacy'
    weight_decay: float = 0.0
    record_federation: bool = True


@app.post('/api/train')
def start_training(request: TrainingRequest):
    cfg = RunConfig(**request.model_dump())
    try:
        cfg.validate()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    manifest = read_json(DATA_DIR / 'manifest.json', {})
    if manifest.get('status') not in ('complete', 'complete_with_failures'):
        raise HTTPException(status_code=409, detail='Wait for the six-site public data download to finish')
    if any(r['status'] in ('preparing', 'running') for r in _runs()):
        raise HTTPException(status_code=409, detail='A real experiment is already running; stop it or wait before starting another')
    if not _training_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail='An experiment is already running; stop it or wait before starting another')
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '-' + cfg.method
    event = threading.Event()
    def execute():
        try:
            run_training(cfg, run_id=run_id, stop_event=event)
        except Exception:
            # Runner writes a detailed persisted failed result.
            import traceback
            traceback.print_exc()
        finally:
            _training_lock.release()
    worker = threading.Thread(target=execute, name='fedbrain-' + run_id, daemon=True)
    _active.update(id=run_id, thread=worker, stop=event)
    worker.start()
    return {'run_id': run_id, 'status': 'preparing'}


@app.post('/api/runs/{run_id}/stop')
def stop(run_id: str):
    directory = _safe_run(run_id)
    result = read_json(directory / 'result.json', {})
    if result.get('status') not in ('preparing', 'running'):
        raise HTTPException(status_code=409, detail='That experiment is not currently running')
    if _active['id'] == run_id and _active['thread'] is not None and _active['thread'].is_alive():
        _active['stop'].set()
    else:
        atomic_json(directory / 'stop_requested.json', {'requested_at': datetime.now().isoformat(), 'source': 'dashboard'})
    return {'status': 'stopping'}


@app.get('/api/protocol')
def protocol():
    return FileResponse(ROOT / 'docs' / 'EXPERIMENT_PLAN.md', media_type='text/markdown; charset=utf-8')


@app.get('/')
def index():
    return FileResponse(ROOT / 'web' / 'index.html')


if (ROOT / 'web').is_dir():
    app.mount('/', StaticFiles(directory=ROOT / 'web'), name='web')


def main():
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='info')


if __name__ == '__main__':
    main()
