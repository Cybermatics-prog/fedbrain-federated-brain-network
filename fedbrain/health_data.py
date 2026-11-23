# SPDX-License-Identifier: MIT
"""Descriptive ABIDE brain-health data views, independent of trained models.

All matrices and curves come from validated real input files. HC/ASD are
research cohort labels, not clinical diagnoses or individual risk estimates.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading

import numpy as np

from .data import (N_ROIS, PHENOTYPE_URL, SITE_ORDER, TIMESERIES_URL,
                   correlation_matrix, load_subjects, read_timeseries)

ROI_LABELS = [f'AAL-{index + 1:03d}' for index in range(N_ROIS)]
ROI_NOTE = 'File column order; the network layout is illustrative, not anatomical coordinates'
_TRIANGLE = np.triu_indices(N_ROIS, 1)
_CACHE_VERSION = 1


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _age(value):
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) and result >= 0 else None


def _mean_absolute(matrix):
    return None if matrix is None else float(np.abs(matrix[_TRIANGLE]).mean())


class HealthDataStore:
    """Lazy, thread-safe, source-fingerprinted full-series FC cache.

    The first request builds missing raw FC matrices. Unchanged subjects are
    reused across disk-cache invalidation; subsequent requests use memory. No
    checkpoint, classifier, stochastic generator or network request is used.
    """

    def __init__(self, data_dir, cache_dir):
        self.data_dir = Path(data_dir).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.lock = threading.RLock()
        self._fingerprint = None
        self._subjects, self._fc, self._n_frames = [], None, None

    def _ensure(self):
        with self.lock:
            subjects = load_subjects(self.data_dir)
            if not subjects:
                raise ValueError('No validated real subjects in the data manifest')
            records = []
            for subject in subjects:
                subject['age'] = _age(subject.get('age'))
                path = Path(subject['path']).resolve()
                stat = path.stat()
                records.append({'id': str(subject['id']), 'file_id': subject['file_id'],
                                'site': subject['site'], 'label': int(subject['label']), 'age': subject['age'],
                                'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
            manifest = self.data_dir / 'manifest.json'
            metadata = {'version': _CACHE_VERSION, 'records': records,
                        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                        'fc': 'fullseries_Pearson_clip1e-6_diagonal1'}
            fingerprint = _digest(metadata)
            if fingerprint == self._fingerprint:
                return
            old_records, old_fc, old_frames = {}, None, None
            cache = self.cache_dir / 'raw-fc-v1.npz'
            if cache.is_file():
                try:
                    with np.load(cache, allow_pickle=False) as stored:
                        old_meta = json.loads(str(stored['metadata'].item()))
                        old_fc, old_frames = stored['fc'], stored['n_frames']
                    if old_meta.get('version') != _CACHE_VERSION or old_meta.get('fc') != metadata['fc']:
                        raise ValueError('Old FC preprocessing version')
                    count = len(old_meta['records'])
                    if old_fc.shape != (count, N_ROIS, N_ROIS) or old_fc.dtype != np.float32 or old_frames.shape != (count,):
                        raise ValueError('Invalid FC cache dimensions')
                    if not np.isfinite(old_fc).all() or np.any(np.abs(old_fc) > 1) or np.any(old_frames < 32):
                        raise ValueError('Invalid FC cache values')
                    if not np.all(np.diagonal(old_fc, axis1=1, axis2=2) == 1):
                        raise ValueError('Invalid FC diagonal')
                    old_records = {row['id']: (index, row) for index, row in enumerate(old_meta['records'])}
                except (ValueError, OSError, KeyError, json.JSONDecodeError):
                    old_records, old_fc, old_frames = {}, None, None
            matrices = np.empty((len(subjects), N_ROIS, N_ROIS), dtype=np.float32)
            frames = np.empty(len(subjects), dtype=np.int64)
            rebuilt = False
            for index, (subject, record) in enumerate(zip(subjects, records)):
                previous = old_records.get(record['id'])
                if previous is not None and previous[1] == record:
                    matrices[index], frames[index] = old_fc[previous[0]], old_frames[previous[0]]
                else:
                    series = read_timeseries(subject['path'])
                    matrix = correlation_matrix(series, fisher=False, clip=1e-6)
                    np.fill_diagonal(matrix, 1.)
                    matrices[index], frames[index] = matrix, len(series)
                    rebuilt = True
            if rebuilt or not cache.is_file() or not old_records or old_meta != metadata:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.tmp')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, fc=matrices, n_frames=frames,
                                        metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
                os.replace(temporary, cache)
            self._subjects, self._fc, self._n_frames = subjects, matrices, frames
            self._fingerprint = fingerprint

    def _mean_fc(self, indices):
        if not len(indices):
            return None
        values = np.asarray(self._fc[indices], dtype=np.float64)
        values = np.arctanh(np.clip(values, -1 + 1e-6, 1 - 1e-6))
        mean = np.tanh(values.mean(axis=0))
        np.fill_diagonal(mean, 1.)
        return mean

    def _subject_row(self, index):
        subject = self._subjects[index]
        return {'id': str(subject['id']), 'site': subject['site'], 'label': int(subject['label']),
                'label_name': 'ASD' if subject['label'] else 'HC', 'age': subject['age'],
                'n_frames': int(self._n_frames[index])}

    @staticmethod
    def _bounds(site, age_min, age_max):
        if site in (None, '', 'All', 'all'):
            site = None
        if site is not None and site not in SITE_ORDER:
            raise ValueError('Unknown site')
        bounds = []
        for value in (age_min, age_max):
            if value is None or value == '':
                bounds.append(None)
                continue
            if isinstance(value, bool):
                raise ValueError('Age bounds must be numeric')
            number = float(value)
            if not np.isfinite(number) or number < 0:
                raise ValueError('Age bounds must be finite and nonnegative')
            bounds.append(number)
        if bounds[0] is not None and bounds[1] is not None and bounds[0] > bounds[1]:
            raise ValueError('age_min exceeds age_max')
        return site, *bounds

    def overview(self, site=None, age_min=None, age_max=None):
        site, age_min, age_max = self._bounds(site, age_min, age_max)
        self._ensure()
        with self.lock:
            selected = []
            for index, subject in enumerate(self._subjects):
                age = subject['age']
                if site is not None and subject['site'] != site:
                    continue
                if (age_min is not None or age_max is not None) and age is None:
                    continue
                if age_min is not None and age < age_min:
                    continue
                if age_max is not None and age > age_max:
                    continue
                selected.append(index)
            hc = [i for i in selected if self._subjects[i]['label'] == 0]
            asd = [i for i in selected if self._subjects[i]['label'] == 1]
            ages = [self._subjects[i]['age'] for i in selected if self._subjects[i]['age'] is not None]
            age_summary = {'count': len(ages), 'missing': len(selected) - len(ages),
                           'min': min(ages) if ages else None, 'max': max(ages) if ages else None,
                           'mean': float(np.mean(ages)) if ages else None,
                           'median': float(np.median(ages)) if ages else None,
                           'range': [min(ages), max(ages)] if ages else None}
            global_ages = [s['age'] for s in self._subjects if s['age'] is not None]
            start = np.floor(min(global_ages) / 5) * 5 if global_ages else 0.
            stop = np.ceil(max(global_ages) / 5) * 5 if global_ages else 5.
            if stop <= start:
                stop = start + 5
            edges = np.arange(start, stop + 5, 5, dtype=np.float64)
            histogram = {'edges': edges.tolist(),
                         'labels': [f'{a:g}–{b:g}' for a, b in zip(edges[:-1], edges[1:])]}
            groups, means = {}, {}
            for name, indices in (('HC', hc), ('ASD', asd)):
                mean = self._mean_fc(indices)
                means[name] = mean
                known = [self._subjects[i]['age'] for i in indices if self._subjects[i]['age'] is not None]
                groups[name] = {'count': len(indices), 'mean_fc': None if mean is None else mean.tolist(),
                                'mean_abs_connectivity': _mean_absolute(mean),
                                'age_mean': float(np.mean(known)) if known else None}
                histogram[name.lower()] = np.histogram(known, bins=edges)[0].tolist()
                histogram['missing_' + name.lower()] = len(indices) - len(known)
            difference = None if means['HC'] is None or means['ASD'] is None else means['ASD'] - means['HC']
            counts = [{'id': name, 'hc': sum(self._subjects[i]['site'] == name for i in hc),
                       'asd': sum(self._subjects[i]['site'] == name for i in asd),
                       'total': sum(self._subjects[i]['site'] == name for i in selected)} for name in SITE_ORDER]
            return {'schema_version': 1,
                    'source': {'name': 'ABIDE-I PCP', 'pipeline': 'CPAC', 'strategy': 'filt_noglobal',
                               'atlas': 'AAL116', 'phenotype_url': PHENOTYPE_URL,
                               'timeseries_url_template': TIMESERIES_URL, 'cohort_fingerprint': self._fingerprint,
                               'aggregation': 'equal-subject Fisher-z mean',
                               'note': 'The research control label HC is not a clinical certification of health; no trained model or risk score is used.'},
                    'filters': {'site': site or 'All', 'age_min': age_min, 'age_max': age_max},
                    'total_subjects': len(self._subjects), 'selected_subjects': len(selected),
                    'n_hc': len(hc), 'n_asd': len(asd), 'age_summary': age_summary,
                    'site_counts': counts, 'age_histogram': histogram, 'groups': groups,
                    'difference': {'count_hc': len(hc), 'count_asd': len(asd),
                                   'fc': None if difference is None else difference.tolist(),
                                   'note': 'Descriptive group difference, unadjusted for confounding, not a disease marker'},
                    'subjects': [self._subject_row(i) for i in selected],
                    'roi_labels': list(ROI_LABELS), 'roi_note': ROI_NOTE}

    def site_previews(self):
        """Actual public-data examples for six site containers, not uploads.

        Select the first HC by anonymous ID (otherwise the first real subject).
        Use cached full individual Pearson FC and one genuine standardized BOLD
        ROI, sampled at at most128 original frames without interpolation.
        """
        self._ensure()
        with self.lock:
            sites = []
            for site in SITE_ORDER:
                indices = [i for i, subject in enumerate(self._subjects) if subject['site'] == site]
                controls = [i for i in indices if self._subjects[i]['label'] == 0]
                example = None
                if indices:
                    chosen = min(controls or indices, key=lambda i: str(self._subjects[i]['id']))
                    subject = self._subjects[chosen]
                    series = read_timeseries(subject['path'])
                    deviations = np.std(series, axis=0)
                    roi = int(np.lexsort((np.arange(N_ROIS), -deviations))[0])
                    values = series[:, roi] - series[:, roi].mean()
                    values = values / deviations[roi] if deviations[roi] > 0 else np.zeros_like(values)
                    frames = np.linspace(0, len(series) - 1, min(128, len(series)), dtype=np.int64)
                    example = {'subject_id': str(subject['id']),
                               'label_name': 'ASD' if subject['label'] else 'HC',
                               'n_frames': len(series), 'fc': self._fc[chosen].tolist(),
                               'bold': {'roi': roi, 'label': ROI_LABELS[roi],
                                        'frame_indices': frames.tolist(), 'values': values[frames].tolist(),
                                        'normalization': 'within-subject full-series ROI zscore before display sampling',
                                        'x_unit': 'frame index, not seconds; TR not available',
                                        'downsampled': len(series) > len(frames)}}
                sites.append({'id': site, 'total': len(indices), 'hc': len(controls),
                              'asd': len(indices) - len(controls), 'example': example})
            return {'schema_version': 1, 'total_subjects': len(self._subjects),
                    'source': {'name': 'ABIDE-I PCP', 'pipeline': 'CPAC', 'strategy': 'filt_noglobal',
                               'atlas': 'AAL116', 'cohort_fingerprint': self._fingerprint},
                    'sites': sites, 'roi_note': ROI_NOTE,
                    'note': 'Public-data examples, not training uploads; they do not imply that the example subject took part in the selected training run.'}

    def subject(self, subject_id):
        self._ensure()
        with self.lock:
            index = next((i for i, s in enumerate(self._subjects) if str(s['id']) == str(subject_id)), None)
            if index is None:
                raise KeyError('Unknown real subject')
            subject = self._subjects[index]
            series = read_timeseries(subject['path'])
            matrix = np.asarray(self._fc[index], dtype=np.float64)
            deviations = np.std(series, axis=0)
            rois = np.lexsort((np.arange(N_ROIS), -deviations))[:3]
            signals = []
            for roi in rois:
                values = series[:, roi] - series[:, roi].mean()
                values = values / deviations[roi] if deviations[roi] > 0 else np.zeros_like(values)
                signals.append({'roi': int(roi), 'label': ROI_LABELS[roi], 'values': values.tolist()})
            scores = np.abs(matrix[_TRIANGLE])
            top = np.lexsort((np.arange(len(scores)), -scores))[:30]
            hc = [i for i, s in enumerate(self._subjects) if s['label'] == 0 and i != index]
            reference = self._mean_fc(hc)
            delta = None if reference is None else matrix - reference
            return {**self._subject_row(index), 'fc': matrix.tolist(), 'roi_labels': list(ROI_LABELS),
                    'roi_note': ROI_NOTE,
                    'top_connections': [{'source': int(_TRIANGLE[0][i]), 'target': int(_TRIANGLE[1][i]),
                                         'value': float(matrix[_TRIANGLE[0][i], _TRIANGLE[1][i]])} for i in top],
                    'signals': {'frames': list(range(len(series))), 'series': signals,
                                'normalization': 'within-subject ROI zscore',
                                'x_unit': 'frame index, not seconds; TR not available'},
                    'mean_abs_connectivity': _mean_absolute(matrix),
                    'reference': {'hc_count': len(hc),
                                  'healthy_mean_fc': None if reference is None else reference.tolist(),
                                  'difference_fc': None if delta is None else delta.tolist(),
                                  'leave_one_out': subject['label'] == 0,
                                  'note': 'Descriptive full-cohort HC reference with the HC subject itself excluded; not a standardised clinical range or a patient risk score.'},
                    'note': 'Research data display only; HC/ASD are source research labels, not a current clinical diagnosis.'}
