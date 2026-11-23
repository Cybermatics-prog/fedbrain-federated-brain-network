# SPDX-License-Identifier: MIT
"""Reproducible single-machine, six-client FedBrain training and inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from .data import (SITE_ORDER, create_subject_splits, ensure_data, features_for_subject,
                   get_fc, load_subjects)
from .metrics import evaluator_self_check, grouped_metrics
from .model import FedBrainModel, domain_alignment_loss, domain_moments
from .privacy import PackedPaillierAggregator, average_shared_states, perturb_shared_state
from .sparse import fit_sparse
from .selection import ValidationCheckpoint, scaled_alignment_loss

ROOT = Path(__file__).resolve().parent.parent
_WRITE_LOCK = threading.Lock()


def portable_path(value):
    """Return a project-relative POSIX path so run records carry no machine paths."""
    try:
        return Path(value).resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        return Path(value).name


def portable_config(config):
    """Configuration as recorded on disk: identical fields, portable directory values."""
    recorded = asdict(config)
    for key in ('data_dir', 'run_root'):
        if isinstance(recorded.get(key), str):
            recorded[key] = portable_path(recorded[key])
    return recorded


def resolve_recorded_path(value, fallback):
    """Resolve a path stored in a run record; relative paths are relative to the repo.

    The result is returned even when it does not exist, so callers can decide what to do;
    an absent override falls back to the default directory.
    """
    if not value:
        return Path(fallback)
    path = Path(value)
    return path if path.is_absolute() else ROOT / path



def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)


def read_json(path, default=None):
    # utf-8-sig tolerates the byte-order mark that Windows editors and PowerShell add.
    try:
        with Path(path).open(encoding='utf-8-sig') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


@dataclass
class RunConfig:
    method: str = 'fedbrain'
    rounds: int = 50
    seed: int = 42
    learning_rate: float = .001
    local_epochs: int = 1
    batch_size: int = 256
    hidden_dim: int = 8
    noise_scale: float = .01
    alignment_weight: float = .1
    max_per_site: int = 0
    max_windows: int = 0
    max_features: int = 512
    sparse_max_iter: int = 40
    sparse_windows_per_subject: int = 8
    lambda_aug: float = .01
    coupling: float = 1.0
    homomorphic: bool = False
    paillier_bits: int = 2048
    clip_norm: float = 10.0
    fold: int | None = None
    n_splits: int = 5
    test_size: float = .2
    validation_fraction: float = .2
    device: str = 'auto'
    data_dir: str = str(ROOT / 'data' / 'abide')
    run_root: str = str(ROOT / 'runs')
    feature_run_id: str | None = None
    checkpoint_selection: str = 'final'
    defer_test: bool = False
    alignment_normalization: str = 'legacy'
    alignment_reference: str = 'legacy'
    weight_decay: float = 0.0
    experiment_tag: str | None = None
    record_federation: bool = True

    def validate(self):
        if self.method not in ('fedbrain', 'fedavg', 'no_alignment', 'no_personalization'):
            raise ValueError('Unknown training method')
        if not 1 <= self.rounds <= 500 or not 1 <= self.local_epochs <= 50:
            raise ValueError('Rounds must be1..500 and local_epochs1..50')
        if not 0 <= self.seed <= 2147483647:
            raise ValueError('Seed must be0..2147483647')
        if self.feature_run_id is not None:
            import re
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,120}', self.feature_run_id):
                raise ValueError('Invalid feature source run identifier')
        if not 0 <= self.noise_scale <= 1 or not 0 <= self.alignment_weight <= 100:
            raise ValueError('Invalid noise/alignment setting')
        if not 1e-6 <= self.learning_rate <= 1 or not 1 <= self.batch_size <= 8192:
            raise ValueError('Invalid learning rate / batch size')
        if not 0 <= self.max_per_site <= 1000 or self.max_windows < 0 or self.max_features < 0:
            raise ValueError('Invalid data/feature budget')
        if not .05 <= self.test_size <= .5 or not .05 <= self.validation_fraction <= .4:
            raise ValueError('Invalid split fractions')
        if self.homomorphic and self.paillier_bits < 2048:
            raise ValueError('Training demonstration requires Paillier>=2048 bits')
        if self.checkpoint_selection not in ('final', 'best_val'):
            raise ValueError('checkpoint_selection must be final or best_val')
        if self.alignment_normalization not in ('legacy', 'batch'):
            raise ValueError('Unknown alignment normalization')
        if self.alignment_reference not in ('legacy', 'post_aggregation'):
            raise ValueError('Unknown alignment reference policy')
        if not 0 <= self.weight_decay <= 1:
            raise ValueError('weight_decay must be0..1')
        if not isinstance(self.record_federation, bool):
            raise ValueError('record_federation must be boolean')
        return self


def _subject_features(subject, config):
    """Use source-fingerprint-validated FC cache owned by the data module."""
    return features_for_subject(subject, window=32, stride=1, max_windows=config.max_windows or None)


def _cap_subjects(subjects, config):
    if not config.max_per_site:
        return subjects
    chosen = []
    for site in SITE_ORDER:
        group = [s for s in subjects if s['site'] == site]
        if len(group) > config.max_per_site:
            group, _ = train_test_split(group, train_size=config.max_per_site, random_state=config.seed,
                                        stratify=[s['label'] for s in group])
        chosen.extend(group)
    return chosen


def _split_with_validation(subjects, config):
    outer = create_subject_splits(subjects, seed=config.seed, test_size=config.test_size,
                                 fold=config.fold, n_splits=config.n_splits)
    train, val = [], []
    for site in SITE_ORDER:
        site_subjects = [s for s in outer['train'] if s['site'] == site]
        if len(site_subjects) < 6:
            raise ValueError(f'{site}: at least6 outer-train subjects required; download more data')
        a, b = train_test_split(site_subjects, test_size=config.validation_fraction,
                                random_state=config.seed + 17, stratify=[s['label'] for s in site_subjects])
        train.extend(a)
        val.extend(b)
    split = {'train': train, 'val': val, 'test': outer['test']}
    sets = [set(s['id'] for s in split[k]) for k in ('train', 'val', 'test')]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i+1, 3)):
        raise AssertionError('Subject leakage detected')
    return split


def _cpu_moments(moments):
    return {'marginal': moments['marginal'].detach().cpu(),
            'conditional': {int(k): v.detach().cpu() for k, v in moments['conditional'].items()},
            'counts': moments.get('counts', {})}


def _reference_moments(models, train_data, device):
    """Synchronized local TRAIN moments in the currently broadcast representation."""
    references = {}
    with torch.inference_mode():
        for site, (x, y) in train_data.items():
            models[site].eval()
            ix = np.linspace(0, len(x)-1, min(2048, len(x)), dtype=int)
            out = models[site](torch.as_tensor(x[ix], device=device))
            references[site] = _cpu_moments(domain_moments(out['hidden'], torch.as_tensor(y[ix], device=device)))
    return references


def _predict_subject(model, x, device, personalization=True):
    probs, global_probs, private_probs, gates = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), 1024):
            out = model(torch.as_tensor(np.array(x[start:start+1024], copy=True), device=device))
            logit = out['mixed_logit'] if personalization else out['global_logit']
            probs.extend(torch.sigmoid(logit).cpu().tolist())
            global_probs.extend(torch.sigmoid(out['global_logit']).cpu().tolist())
            private_probs.extend(torch.sigmoid(out['private_logit']).cpu().tolist())
            gates.extend(out['gate'].cpu().tolist())
    return {'asd_probability': float(np.mean(probs)), 'global_probability': float(np.mean(global_probs)),
            'private_probability': float(np.mean(private_probs)), 'gate': float(np.mean(gates)) if personalization else 1.,
            'predicted_label': int(np.mean(probs) >= .5), 'window_probabilities': probs,
            'n_windows': len(probs)}


def _evaluate(subjects, matrices, models, device, personalization):
    result = []
    for s in subjects:
        pred = _predict_subject(models[s['site']], matrices[s['id']], device, personalization)
        result.append({**{k: s[k] for k in ('id', 'site', 'label')}, **pred})
    return result


def run_training(config: RunConfig, run_id=None, stop_event=None, progress=None):
    config.validate()
    run_id = run_id or datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + config.method
    directory = Path(config.run_root) / run_id
    directory.mkdir(parents=True, exist_ok=True)
    status = {'id': run_id, 'status': 'preparing', 'created_at': datetime.now(timezone.utc).isoformat(),
              'config': portable_config(config), 'history': [], 'round': 0, 'total_rounds': config.rounds,
              'process_id': os.getpid(),
              'privacy': {'noise_scale': config.noise_scale, 'homomorphic': config.homomorphic,
                          'formal_dp': False, 'threat_model': 'single-process simulator'},
              'test_policy': ('DEFERRED: validation-only search, no test predictions' if config.defer_test else
                              'test evaluated once after fixed training budget and configured checkpoint selection'),
              'test_evaluated': False,
              'checkpoint_selection': config.checkpoint_selection,
              'metrics': None, 'stage': 'loading_data',
              'environment': {'torch': torch.__version__, 'numpy': np.__version__},
              'source_hashes': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                for name in (('runner.py','model.py','sparse.py','selection.py','privacy.py') +
                                             (('federation.py',) if config.record_federation else ()))}}
    recorder = None
    def save():
        status['updated_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(directory / 'result.json', status)
        if progress:
            progress(status)
    def stopped():
        return (stop_event is not None and stop_event.is_set()) or (directory / 'stop_requested.json').exists()
    def cancel_if_requested():
        if stopped():
            raise InterruptedError('Stop requested by user')
    save()
    started = time.perf_counter()
    try:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        torch.use_deterministic_algorithms(True, warn_only=True)
        rng = np.random.default_rng(config.seed)
        status['evaluator_self_check'] = evaluator_self_check()
        subjects = _cap_subjects(load_subjects(config.data_dir), config)
        if not subjects:
            raise ValueError('No real data. Run python -m fedbrain.runner download first')
        split = _split_with_validation(subjects, config)
        atomic_json(directory / 'split.json', {key: [{'id': s['id'], 'site': s['site'], 'label': s['label']} for s in values]
                                               for key, values in split.items()})
        status['split_counts'] = {key: len(values) for key, values in split.items()}
        status['data_subjects'] = len(subjects)
        status['stage'] = 'dynamic_connectivity'
        save()
        # Moments, feature selection and transform fitted on TRAIN ONLY.
        sums = np.zeros(6670, dtype=np.float64)
        sums2 = np.zeros(6670, dtype=np.float64)
        count = 0
        sparse_site = {site: {'X': [], 'y': [], 'segments': [], 'subject_ids': []} for site in SITE_ORDER}
        for index, s in enumerate(split['train']):
            cancel_if_requested()
            x = _subject_features(s, config)
            sums += np.sum(x, axis=0, dtype=np.float64)
            sums2 += np.sum(np.asarray(x, dtype=np.float64)**2, axis=0)
            count += len(x)
            idx = np.linspace(0, len(x)-1, min(len(x), config.sparse_windows_per_subject), dtype=int)
            obj = sparse_site[s['site']]
            obj['X'].append(np.asarray(x[idx], dtype=np.float32))
            obj['y'].extend([s['label']] * len(idx))
            obj['segments'].extend(np.arange(len(idx)).tolist())
            obj['subject_ids'].extend([s['id']] * len(idx))
            if index % 20 == 0:
                status['progress_detail'] = f'Training-subject connectivity matrices {index+1}/{len(split["train"])}'
                save()
        mean = sums / count
        std = np.sqrt(np.maximum(sums2 / count - mean**2, 1e-6))
        mean, std = mean.astype(np.float32), std.astype(np.float32)
        for obj in sparse_site.values():
            obj['X'] = (np.concatenate(obj['X']) - mean) / std
            obj['y'] = np.asarray(obj['y'], dtype=np.int64)
            obj['segments'] = np.asarray(obj['segments'], dtype=np.int64)
        status['stage'] = 'sparse_selection'
        save()
        if config.feature_run_id:
            prior_path = Path(config.run_root) / config.feature_run_id / 'checkpoint.pt'
            if prior_path.is_file():
                prior = torch.load(prior_path, map_location='cpu', weights_only=False)
                prior_split = read_json(prior_path.parent / 'split.json')
                now_ids = sorted(s['id'] for s in split['train'])
                if sorted(s['id'] for s in prior_split['train']) != now_ids:
                    raise ValueError('Cannot reuse feature selection from different training subjects')
                indices = np.array(prior['indices'], dtype=np.int64)
                selection = {'reused_from': config.feature_run_id, 'train_only': True}
            else:
                # A clean clone has no reference checkpoint. Refit here rather than fail, and
                # record that this run is not the reused-selection configuration.
                print(f'warning: {prior_path} is absent; fitting the sparse selection in this run',
                      flush=True)
                indices, fitted = fit_sparse(sparse_site, lambda_aug=config.lambda_aug,
                                             coupling=config.coupling, max_iter=config.sparse_max_iter,
                                             max_features=config.max_features or None, seed=config.seed)
                selection = {'reused_from': None, 'train_only': True,
                             'requested_feature_run_id': config.feature_run_id,
                             'fallback': 'requested_feature_run_missing',
                             'note': ('Sparse selection was fitted by this run because the requested '
                                      'feature run is absent. Structure matches the recorded demo trace; '
                                      'numeric values do not.'),
                             **fitted}
        else:
            indices, selection = fit_sparse(sparse_site, lambda_aug=config.lambda_aug,
                                             coupling=config.coupling, max_iter=config.sparse_max_iter,
                                             max_features=config.max_features or None, seed=config.seed)
        del sparse_site
        indices = np.asarray(indices, dtype=np.int64)
        if not len(indices):
            raise ValueError('Sparse optimizer selected no features; no test-informed fallback allowed')
        status['selection'] = {'n_selected': len(indices), 'total_features': 6670,
                               'max_features': config.max_features, 'diagnostics': selection,
                               'reused_from': selection.get('reused_from'),
                               'fallback': selection.get('fallback'),
                               'sparse_windows_per_subject': config.sparse_windows_per_subject,
                               'segment_grid': 'relative-time evenly spaced bins; explicit sparse-stage approximation',
                               'subject_cap': 'seeded within-site stratified sample' if config.max_per_site else None}
        np.savez(directory / 'preprocessing.npz', mean=mean, std=std, indices=indices)
        matrices = {}
        processing_subjects = split['train'] + split['val'] if config.defer_test else subjects
        for i, s in enumerate(processing_subjects):
            cancel_if_requested()
            x = _subject_features(s, config)
            matrices[s['id']] = np.asarray((x[:, indices] - mean[indices]) / std[indices], dtype=np.float32)
            if i % 30 == 0:
                status['progress_detail'] = f'Selected feature transforms {i+1}/{len(subjects)}'
                save()
        train_data = {}
        for site in SITE_ORDER:
            site_subjects = [s for s in split['train'] if s['site'] == site]
            train_data[site] = (np.concatenate([matrices[s['id']] for s in site_subjects]),
                                np.concatenate([np.full(len(matrices[s['id']]), s['label'], dtype=np.float32)
                                                for s in site_subjects]))
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if config.device == 'auto' else config.device
        status['device'] = device
        base = FedBrainModel(len(indices), config.hidden_dim)
        initial_shared = base.shared_state_dict()
        models = {site: FedBrainModel(len(indices), config.hidden_dim).to(device) for site in SITE_ORDER}
        for m in models.values():
            m.load_shared_state_dict(initial_shared)
        use_personal = config.method not in ('fedavg', 'no_personalization')
        use_align = config.method not in ('fedavg', 'no_alignment') and config.alignment_weight > 0
        reference = {}
        if use_align:
            with torch.inference_mode():
                for site, (x, y) in train_data.items():
                    ix = np.linspace(0, len(x)-1, min(2048, len(x)), dtype=int)
                    out = models[site](torch.as_tensor(x[ix], device=device))
                    reference[site] = _cpu_moments(domain_moments(out['hidden'], torch.as_tensor(y[ix], device=device)))
        encryptor = PackedPaillierAggregator(config.paillier_bits) if config.homomorphic else None
        total_bytes = 0
        if config.record_federation:
            from .federation import FederationRecorder
            recorder = FederationRecorder(directory, models=models, train_data=train_data,
                config=portable_config(config), train_subject_counts={site:sum(s['site']==site for s in split['train']) for site in SITE_ORDER},
                feature_indices=indices, site_order=SITE_ORDER)
            status['federation'] = {'available': True, 'schema_version': 1, 'current_round': 0,
                                    'current_phase': 'ready', 'current_site': None, 'completed_rounds': 0}
        selector = ValidationCheckpoint()
        status.update(status='running', stage='training', progress_detail='')
        save()
        for round_index in range(1, config.rounds+1):
            cancel_if_requested()
            if recorder is not None:
                recorder.start_round(round_index, models)
                status['federation'].update(current_round=round_index, current_phase='ready', current_site=None)
            losses, alignments, weighted_alignments, states, next_reference = [], [], [], [], {}
            for site in SITE_ORDER:
                cancel_if_requested()
                model = models[site]
                if recorder is not None:
                    recorder.client_started(site)
                    status['federation'].update(current_phase='local_train', current_site=site)
                    status['progress_detail'] = f'Round {round_index} · {site} local training (data stays at this site)'
                    save()
                site_loss_start = len(losses)
                model.train()
                # Optimizer state reset per local communication round (explicit implementation choice).
                optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
                x, y = train_data[site]
                for epoch in range(config.local_epochs):
                    order = rng.permutation(len(x))
                    for start in range(0, len(order), config.batch_size):
                        ix = order[start:start+config.batch_size]
                        xb = torch.as_tensor(x[ix], device=device)
                        yb = torch.as_tensor(y[ix], device=device)
                        out = model(xb)
                        logit = out['mixed_logit'] if use_personal else out['global_logit']
                        ce = F.binary_cross_entropy_with_logits(logit, yb)
                        align = (domain_alignment_loss(out['hidden'], yb,
                                 [reference[s] for s in SITE_ORDER if s != site]) if use_align else ce.new_zeros(()))
                        loss, weighted_align = scaled_alignment_loss(ce, align, len(yb),
                            config.alignment_weight, config.alignment_normalization)
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Non-finite training loss')
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                        optimizer.step()
                        losses.append(float(ce.detach()))
                        alignments.append(float(align.detach()))
                        weighted_alignments.append(float(weighted_align.detach()))
                model.eval()
                with torch.inference_mode():
                    ix = np.linspace(0, len(x)-1, min(2048, len(x)), dtype=int)
                    out = model(torch.as_tensor(x[ix], device=device))
                    next_reference[site] = _cpu_moments(domain_moments(out['hidden'], torch.as_tensor(y[ix], device=device)))
                uploaded_state = perturb_shared_state(model.shared_state_dict(), config.noise_scale,
                                                        config.clip_norm, rng)
                states.append(uploaded_state)
                if recorder is not None:
                    recorder.client_uploaded(site, model, uploaded_state,
                        loss=float(np.mean(losses[site_loss_start:])),
                        alignment_loss=float(np.mean(alignments[site_loss_start:])))
                    status['federation'].update(current_phase='upload', current_site=site)
                    save()
            if recorder is not None:
                status['federation'].update(current_phase='aggregate', current_site=None)
                status['progress_detail'] = f'Round {round_index} · server aggregates the six sites\' shared parameters'
                save()
            shared = average_shared_states(states, encrypted=encryptor)
            if recorder is not None:
                recorder.aggregated(shared)
            param_bytes = sum(v.numel() * v.element_size() for v in shared.values())
            # Includes upload+download, not local disk/metadata. HE measured ciphertext upload.
            round_bytes = (encryptor.last_wire_bytes if encryptor else param_bytes * len(SITE_ORDER)) + param_bytes * len(SITE_ORDER)
            # Domain statistics are transmitted separately (marginal and class means).
            if use_align:
                round_bytes += len(SITE_ORDER) * config.hidden_dim * 3 * 4 * 2
            total_bytes += round_bytes
            for model in models.values():
                model.load_shared_state_dict(shared)
            if recorder is not None:
                recorder.broadcast(models)
                status['federation'].update(current_phase='broadcast', current_site=None,
                                            completed_rounds=round_index)
                status['progress_detail'] = f'Round {round_index} · shared model broadcast; private experts retained at this site'
            if config.alignment_reference == 'post_aggregation' and use_align:
                next_reference = _reference_moments(models, train_data, device)
            reference = next_reference
            val_records = _evaluate(split['val'], matrices, models, device, use_personal)
            val_metrics = grouped_metrics(val_records)
            history_row = {'round': round_index, 'train_loss': float(np.mean(losses)),
                           'alignment_loss': float(np.mean(alignments)),
                           'weighted_alignment_loss': float(np.mean(weighted_alignments)),
                           'val_accuracy': val_metrics['site_mean_accuracy'],
                           'val_auc': val_metrics['auc'],
                           'site_accuracy': {s: v['accuracy'] for s, v in val_metrics['per_site'].items()},
                           'elapsed_seconds': time.perf_counter() - started,
                           'communication_bytes': total_bytes}
            if config.checkpoint_selection == 'best_val':
                selector.consider(round_index, val_metrics['site_mean_accuracy'], models)
                status['selected_round'] = selector.round
                status['best_validation_accuracy'] = selector.score
            status['history'].append(history_row)
            status['round'] = round_index
            save()
            print(f"{run_id} round={round_index}/{config.rounds} loss={history_row['train_loss']:.4f} "
                  f"val_macro={history_row['val_accuracy']:.4f}", flush=True)
        cancel_if_requested()
        if config.checkpoint_selection == 'best_val':
            selector.restore(models)
            val_records = _evaluate(split['val'], matrices, models, device, use_personal)
        else:
            status['selected_round'] = config.rounds
        status['stage'] = 'validation_only_evaluation' if config.defer_test else 'heldout_evaluation'
        save()
        train_records = _evaluate(split['train'], matrices, models, device, use_personal)
        status['train_metrics'] = grouped_metrics(train_records)
        status['validation_metrics'] = grouped_metrics(val_records)
        atomic_json(directory / 'validation_predictions.json', val_records)
        if not config.defer_test:
            test_records = _evaluate(split['test'], matrices, models, device, use_personal)
            status['metrics'] = grouped_metrics(test_records)
            status['test_evaluated'] = True
            status['generalization_gap'] = status['train_metrics']['accuracy'] - status['metrics']['accuracy']
            atomic_json(directory / 'predictions.json', test_records)
        checkpoint = {'config': portable_config(config), 'indices': indices, 'mean': mean, 'std': std,
                      'models': {s: {k: v.detach().cpu() for k, v in m.state_dict().items()} for s, m in models.items()},
                      'subject_ids': [s['id'] for s in subjects], 'run_id': run_id,
                      'personalization': use_personal, 'selected_round': status['selected_round']}
        torch.save(checkpoint, directory / 'checkpoint.pt')
        status.update(status='completed', stage='completed', elapsed_seconds=time.perf_counter()-started,
                      communication_bytes=total_bytes)
        if recorder is not None:
            recorder.index['metadata']['checkpoint_selection'] = config.checkpoint_selection
            recorder.index['metadata']['checkpoint_selected_round'] = status['selected_round']
            recorder.finish(status='completed')
            status['federation'].update(current_phase='completed', current_site=None)
        save()
        return status
    except InterruptedError as exc:
        if recorder is not None:
            recorder.finish(status='stopped')
        status.update(status='stopped', stage='stopped', error=str(exc), elapsed_seconds=time.perf_counter()-started)
        save()
        return status
    except Exception as exc:
        if recorder is not None:
            try:
                recorder.finish(status='failed')
            except Exception as trace_exc:
                status['federation_error'] = str(trace_exc)
        status.update(status='failed', stage='failed', error=str(exc), traceback=traceback.format_exc(),
                      elapsed_seconds=time.perf_counter()-started)
        save()
        raise


def infer_subject(run_dir, subject_id, subjects=None):
    run_dir = Path(run_dir)
    checkpoint = torch.load(run_dir / 'checkpoint.pt', map_location='cpu', weights_only=False)
    config = RunConfig(**checkpoint['config'])
    config.data_dir = str(resolve_recorded_path(config.data_dir, ROOT / 'data' / 'abide'))
    config.run_root = str(resolve_recorded_path(config.run_root, ROOT / 'runs'))
    if not (Path(config.data_dir) / 'manifest.json').exists():
        config.data_dir = str(ROOT / 'data' / 'abide')
    subjects = subjects or load_subjects(config.data_dir)
    subject = next((s for s in subjects if str(s['id']) == str(subject_id)), None)
    if subject is None:
        raise KeyError('Unknown real subject')
    x = _subject_features(subject, config)
    ix = checkpoint['indices']
    x = np.asarray((x[:, ix] - checkpoint['mean'][ix]) / checkpoint['std'][ix], dtype=np.float32)
    model = FedBrainModel(len(ix), config.hidden_dim)
    model.load_state_dict(checkpoint['models'][subject['site']])
    prediction = _predict_subject(model, x, 'cpu', checkpoint['personalization'])
    fc = np.tanh(get_fc(subject))  # Display Pearson r; train on Fisher-z.
    rows, cols = np.triu_indices(116, 1)
    top = np.argsort(-np.abs(fc[rows, cols]))[:80]
    split = read_json(run_dir / 'split.json', {})
    split_name = next((k for k, values in split.items() if any(str(s['id']) == str(subject_id) for s in values)), 'unseen')
    return {**{k: subject.get(k) for k in ('id', 'site', 'label', 'age')},
            'label_name': 'ASD' if subject['label'] else 'HC', 'split': split_name,
            'fc': np.asarray(fc).round(4).tolist(), 'prediction': prediction,
            'top_connections': [{'source': int(rows[i]), 'target': int(cols[i]), 'value': float(fc[rows[i], cols[i]])}
                                for i in top],
            'roi_labels': [f'AAL-{i+1:03d}' for i in range(116)],
            'note': 'Real ABIDE subject. FC is full-series Pearson correlation; prediction uses 32-frame sliding-window Fisher-z features. '
                    'AAL indices follow file column order and the network coordinates are an illustrative layout, not anatomical positions. Research demo only; not for clinical diagnosis.'}


def main():
    parser = argparse.ArgumentParser(description='FedBrain reproducible research runner')
    sub = parser.add_subparsers(dest='command', required=True)
    d = sub.add_parser('download')
    d.add_argument('--max-per-site', type=int, default=0)
    d.add_argument('--workers', type=int, default=6)
    t = sub.add_parser('train')
    t.add_argument('--config', type=Path)
    t.add_argument('--run-id')
    t.add_argument('--rounds', type=int)
    t.add_argument('--method', choices=['fedbrain', 'fedavg', 'no_alignment', 'no_personalization'])
    t.add_argument('--seed', type=int)
    t.add_argument('--max-per-site', type=int)
    t.add_argument('--fold', type=int)
    t.add_argument('--max-features', type=int)
    t.add_argument('--noise-scale', type=float)
    t.add_argument('--homomorphic', action='store_true', default=None)
    t.add_argument('--feature-run-id')
    args = parser.parse_args()
    if args.command == 'download':
        result = ensure_data(ROOT / 'data' / 'abide', max_per_site=args.max_per_site or None, workers=args.workers,
                             progress=lambda e: print(json.dumps(e, ensure_ascii=False), flush=True))
        summary = {k: v for k, v in result.items() if k not in ('subjects', 'candidates', 'failures')}
        summary['subjects'] = len(result.get('subjects') or [])
        summary['failures'] = len(result.get('failures') or [])
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        if args.config:
            # Never fall back to defaults when an explicit config cannot be read: a silent
            # default run would report a different experiment under the requested run id.
            if not Path(args.config).is_file():
                raise SystemExit(f'config file not found: {args.config}')
            cfg = read_json(args.config)
            if not isinstance(cfg, dict):
                raise SystemExit(f'config file is not valid JSON: {args.config}')
            unknown = sorted(set(cfg) - set(RunConfig.__dataclass_fields__))
            if unknown:
                raise SystemExit('unknown config keys: ' + ', '.join(unknown))
        else:
            cfg = {}
        for name in ('rounds', 'method', 'seed', 'max_per_site', 'fold', 'max_features', 'noise_scale', 'homomorphic', 'feature_run_id'):
            value = getattr(args, name)
            if value is not None:
                cfg[name] = value
        result = run_training(RunConfig(**cfg), args.run_id)
        print(json.dumps({'id': result['id'], 'metrics': result['metrics']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
