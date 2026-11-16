"""Read-only, train-only telemetry for actual federated collaboration events.

No training, RNG calls, raw FC or individual probe points are exported. All
representation summaries use one fixed initial-training PCA coordinate system.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time
from datetime import datetime, timezone

import numpy as np
import torch


class FederationRecorder:
    def __init__(self, directory, *, models, train_data, config,
                 train_subject_counts, feature_indices, site_order):
        self.directory = Path(directory) / 'federation'
        if (self.directory / 'index.json').exists():
            raise FileExistsError('Refusing to overwrite an existing federation trace')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.site_order = list(site_order)
        if not self.site_order or set(self.site_order) != set(models):
            raise ValueError('site_order must identify every client exactly once')
        self.config = dict(config)
        self.models = models
        first = models[self.site_order[0]]
        shared = first.shared_state_dict()
        self.shared_names = list(shared)
        self.shared_shapes = {key: tuple(value.shape) for key, value in shared.items()}
        self.shared_count = sum(value.numel() for value in shared.values())
        if not self.shared_count:
            raise ValueError('No shared parameters')
        self.sketch_coordinates = np.linspace(0, self.shared_count - 1, 64, dtype=np.int64)
        coordinates, offset = [], 0
        for name, tensor in shared.items():
            for coordinate in self.sketch_coordinates:
                if offset <= coordinate < offset + tensor.numel():
                    coordinates.append((int(coordinate), name, list(np.unravel_index(int(coordinate) - offset, tensor.shape))))
            offset += tensor.numel()
        # Preserve the exact linspace order, including repeated indices for tiny
        # unit-test models with fewer than64 shared scalars.
        coordinate_map = {flat: (name, index) for flat, name, index in coordinates}
        coordinate_metadata = [{'flat_index': int(flat), 'parameter': coordinate_map[int(flat)][0],
                                'parameter_index': [int(i) for i in coordinate_map[int(flat)][1]]}
                               for flat in self.sketch_coordinates]
        self.probes, initial_hidden, site_metadata = {}, [], {}
        for site in self.site_order:
            X, labels = train_data[site]
            X, labels = np.asarray(X), np.asarray(labels).reshape(-1)
            if X.ndim != 2 or len(X) == 0 or len(labels) != len(X):
                raise ValueError('Expected nonempty training X and matching y per client')
            indices = np.linspace(0, len(X) - 1, min(128, len(X)), dtype=np.int64)
            self.probes[site] = (np.array(X[indices], copy=True), np.array(labels[indices], copy=True))
            hidden, _ = self._hidden_gate(models[site], site)
            initial_hidden.append(hidden)
            private_count = sum(value.numel() for value in models[site].private_state_dict().values())
            site_metadata[site] = {'train_subject_count': int(train_subject_counts[site]),
                                   'train_window_count': len(X), 'probe_window_count': len(indices),
                                   'private_parameter_count': private_count}
        pooled = np.concatenate(initial_hidden)
        self.center = pooled.mean(axis=0)
        centered = pooled - self.center
        covariance = centered.T @ centered / max(len(pooled) - 1, 1)
        eigenvalues, vectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        self.basis = np.zeros((pooled.shape[1], 2))
        explained = np.zeros(2)
        for component, eigenindex in enumerate(order[:2]):
            vector = vectors[:, eigenindex]
            if vector[np.argmax(np.abs(vector))] < 0:
                vector = -vector
            self.basis[:, component] = vector
            explained[component] = max(float(eigenvalues[eigenindex]), 0)
        denominator = float(np.maximum(eigenvalues, 0).sum())
        ratios = explained / denominator if denominator else np.zeros(2)
        self.index = {
            'available': True, 'schema_version': 1, 'run_id': Path(directory).name,
            'status': 'running', 'site_order': self.site_order,
            'total_rounds': int(config.get('rounds', 0)), 'completed_rounds': 0,
            'current_round': 0, 'current_phase': 'ready', 'current_site': None,
            'metadata': {
                'sites': site_metadata, 'shared_parameter_count': self.shared_count,
                'shared_parameter_names': self.shared_names,
                'private_parameter_count': site_metadata[self.site_order[0]]['private_parameter_count'],
                'selected_feature_count': len(feature_indices),
                'sketch': {'size': 64, 'space': 'shared_model_parameters', 'coordinates': coordinate_metadata},
                'projection': {'method': 'PCA', 'fixed': True,
                               'fit_scope': 'initial_training_probe_hidden_only', 'n_components': 2,
                               'explained_variance_ratio': ratios.tolist(), 'center': self.center.tolist(),
                               'components': self.basis.T.tolist(), 'probe_limit_per_site': 128},
                'upload_bytes_measurement': 'shared_tensor_payload_before_optional_HE_packing_excludes_protocol_overhead',
                'aggregation_distance_source': 'actual_uploaded_shared_states',
                'representation_interpretation': 'local-post-training versus actual post-broadcast; not an isolated MMD causal effect',
                'privacy': 'aggregate train-probe summaries only; private weights never exported; no formal DP claim',
            }, 'events': [], 'rounds': [],
        }
        self.frame = None
        self._event('ready', None, 'recorder_initialized')

    def _atomic(self, path, value):
        temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.{threading.get_ident()}.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
        os.replace(temporary, path)

    def _save(self):
        if self.frame is not None:
            self._atomic(self.directory / f"round-{self.frame['round']:04d}.json", self.frame)
        self._atomic(self.directory / 'index.json', self.index)

    def _event(self, phase, site, event):
        self.index.update(current_phase=phase, current_site=site)
        self.index['events'].append({'sequence': len(self.index['events']) + 1,
                                     'timestamp': datetime.now(timezone.utc).isoformat(),
                                     'monotonic_ns': time.monotonic_ns(),
                                     'round': self.index['current_round'], 'phase': phase,
                                     'site': site, 'event': event})
        self._save()

    def _vector(self, state):
        if set(state) != set(self.shared_names):
            raise ValueError('Shared state keys differ from initial model')
        arrays = []
        for name in self.shared_names:
            value = state[name]
            if tuple(value.shape) != self.shared_shapes[name]:
                raise ValueError('Shared parameter shape changed')
            arrays.append(value.detach().cpu().double().numpy().reshape(-1).copy())
        vector = np.concatenate(arrays)
        if not np.isfinite(vector).all():
            raise ValueError('Non-finite shared parameters')
        return vector

    def _sketch(self, vector):
        return vector[self.sketch_coordinates].tolist()

    @staticmethod
    def _private_hash(model):
        digest = hashlib.sha256()
        for name, tensor in model.private_state_dict().items():
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def _hidden_gate(self, model, site):
        flags = [(module, module.training) for module in model.modules()]
        parameter = next(model.parameters())
        try:
            model.eval()
            inputs = torch.as_tensor(self.probes[site][0], device=parameter.device, dtype=parameter.dtype)
            with torch.inference_mode():
                output = model(inputs)
                hidden = output['hidden'].detach().cpu().double().numpy().copy()
                gate = output['gate'].detach().cpu().double().numpy().copy()
        finally:
            for module, flag in flags:
                module.training = flag
        if not np.isfinite(hidden).all() or not np.isfinite(gate).all():
            raise ValueError('Non-finite probe outputs')
        return hidden, gate

    def _representation(self, model, site):
        hidden, gate = self._hidden_gate(model, site)
        projected = (hidden - self.center) @ self.basis
        mean = projected.mean(axis=0)
        centered = projected - mean
        covariance = centered.T @ centered / max(len(projected) - 1, 1)
        labels = self.probes[site][1]
        classes = {}
        for label in (0, 1):
            mask = labels == label
            if mask.any():
                classes[str(label)] = {'count': int(mask.sum()), 'mean': projected[mask].mean(axis=0).tolist()}
        return {'count': len(projected), 'mean': mean.tolist(), 'covariance': covariance.tolist(),
                'class_centers': classes, 'gate_mean': float(gate.mean())}

    def _site_snapshot(self, site, model):
        return {'shared_sketch': self._sketch(self._vector(model.shared_state_dict())),
                'private_hash': self._private_hash(model), 'representation': self._representation(model, site)}

    def start_round(self, round_index, models):
        with self.lock:
            if self.frame is not None and self.frame['status'] != 'completed':
                raise RuntimeError('Previous round has not been broadcast')
            if int(round_index) <= self.index['current_round']:
                raise ValueError('Round indices must increase')
            self.models = models
            self.incoming = self._vector(models[self.site_order[0]].shared_state_dict())
            incoming_error = max(float(np.max(np.abs(self._vector(models[s].shared_state_dict()) - self.incoming))) for s in self.site_order)
            self.uploads, self.clean_vectors = {}, {}
            self.frame = {'round': int(round_index), 'status': 'in_progress',
                          'before': {'shared_sketch': self._sketch(self.incoming),
                                     'shared_consensus_max_error': incoming_error,
                                     'sites': {s: self._site_snapshot(s, models[s]) for s in self.site_order}},
                          'clients': {}, 'aggregation': None, 'after': None}
            self.index['current_round'] = int(round_index)
            self.index['rounds'].append({'round': int(round_index), 'status': 'in_progress',
                                         'path': f'round-{int(round_index):04d}.json'})
            self._event('local_train', None, 'round_started')

    def client_started(self, site):
        with self.lock:
            if site not in self.site_order or self.frame is None:
                raise ValueError('Unknown client or no active round')
            self._event('local_train', site, 'client_started')

    def client_uploaded(self, site, model, noisy_state, *, loss=None, alignment_loss=None):
        with self.lock:
            if self.frame is None or site not in self.site_order or site in self.uploads:
                raise ValueError('Invalid or duplicate client upload')
            clean = self._vector(model.shared_state_dict())
            uploaded = self._vector(noisy_state)
            norm = float(np.linalg.norm(clean))
            clip = min(1., float(self.config.get('clip_norm', 10.)) / max(norm, 1e-12))
            self.clean_vectors[site], self.uploads[site] = clean, uploaded
            self.frame['clients'][site] = {
                'update_l2': float(np.linalg.norm(clean - self.incoming)),
                'upload_perturbation_l2': float(np.linalg.norm(uploaded - clean)),
                'clip_factor': clip,
                'upload_bytes': int(sum(value.numel() * value.element_size() for value in noisy_state.values())),
                'shared_sketch': self._sketch(uploaded), 'clean_shared_sketch': self._sketch(clean),
                'update_sketch': self._sketch(clean - self.incoming),
                'private_hash': self._private_hash(model), 'representation': self._representation(model, site),
                'loss': None if loss is None else float(loss),
                'alignment_loss': None if alignment_loss is None else float(alignment_loss)}
            self._event('upload', site, 'client_uploaded')

    def aggregated(self, shared):
        with self.lock:
            if set(self.uploads) != set(self.site_order):
                raise RuntimeError('Aggregation must follow all actual client uploads')
            self.aggregate_vector = self._vector(shared)
            expected = np.stack([self.uploads[s] for s in self.site_order]).mean(axis=0)
            self.private_before_broadcast = {s: self._private_hash(self.models[s]) for s in self.site_order}
            self.frame['aggregation'] = {
                'shared_sketch': self._sketch(self.aggregate_vector),
                'server_delta_l2': float(np.linalg.norm(self.aggregate_vector - self.incoming)),
                'mean_local_distance_to_aggregate_l2': float(np.mean([np.linalg.norm(self.uploads[s] - self.aggregate_vector) for s in self.site_order])),
                'mean_clean_local_distance_to_aggregate_l2': float(np.mean([np.linalg.norm(self.clean_vectors[s] - self.aggregate_vector) for s in self.site_order])),
                'equal_mean_max_error': float(np.max(np.abs(expected - self.aggregate_vector)))}
            self._event('aggregate', None, 'shared_aggregated')
            self._event('broadcast', None, 'broadcast_pending')

    def broadcast(self, models):
        with self.lock:
            if self.frame is None or self.frame['aggregation'] is None:
                raise RuntimeError('Broadcast must follow aggregation')
            sites = {s: self._site_snapshot(s, models[s]) for s in self.site_order}
            preserved = {s: sites[s]['private_hash'] == self.private_before_broadcast[s] for s in self.site_order}
            error = max(float(np.max(np.abs(self._vector(models[s].shared_state_dict()) - self.aggregate_vector))) for s in self.site_order)
            self.frame['after'] = {'sites': sites, 'shared_consensus_max_error': error,
                                    'private_preserved': all(preserved.values()),
                                    'private_preserved_by_site': preserved}
            self.frame['status'] = 'completed'
            self.index['rounds'][-1]['status'] = 'completed'
            self.index['completed_rounds'] = sum(row['status'] == 'completed' for row in self.index['rounds'])
            self._event('broadcast', None, 'broadcast_verified')
            self._event('ready', None, 'round_completed')

    def finish(self, status='completed'):
        with self.lock:
            self.index['status'] = str(status)
            if self.frame is not None and self.frame['status'] != 'completed':
                self.frame['status'] = str(status)
                self.index['rounds'][-1]['status'] = str(status)
            self._event('ready', None, 'recorder_finished')
