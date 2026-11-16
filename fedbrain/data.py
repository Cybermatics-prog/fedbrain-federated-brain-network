"""ABIDE-I CPAC/AAL116 acquisition and leakage-safe subject preprocessing.

The default cohort is every available non-FAIL quality-rated subject from the
six paper sites, not a truncation to the paper's undocumented QC counts.
Phenotype labels DX_GROUP=1/2 map to ASD=1/HC=0. Subject splits always precede
sliding-window augmentation. Network access occurs only in ``ensure_data``.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

PHENOTYPE_URL = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv"
TIMESERIES_URL = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_noglobal/rois_aal/{file_id}_rois_aal.1D"
SITE_CODES = {"NYU": "NYU", "UM_1": "UM", "USM": "USM", "UCLA_1": "UCLA", "KKI": "KKI", "YALE": "Yale"}
SITE_ORDER = ("NYU", "UM", "USM", "UCLA", "KKI", "Yale")
PAPER_COUNTS = dict(zip(SITE_ORDER, (167, 88, 52, 63, 48, 56)))
QC_FIELDS = ("qc_rater_1", "qc_anat_rater_2", "qc_func_rater_2", "qc_anat_rater_3", "qc_func_rater_3")
N_ROIS = 116
N_FEATURES = N_ROIS * (N_ROIS - 1) // 2
_TRIANGLE = np.triu_indices(N_ROIS, k=1)
_THREAD_LOCAL = threading.local()
_MANIFEST_VERSION = 1


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def _session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    if not hasattr(_THREAD_LOCAL, "session"):
        session = requests.Session()
        retry = Retry(total=3, connect=3, read=3, status=3, backoff_factor=0.6,
                      status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
        session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=6, pool_maxsize=6))
        _THREAD_LOCAL.session = session
    return _THREAD_LOCAL.session


def _emit(progress, **event):
    if progress is not None:
        progress(event)


def _optional_float(value):
    try:
        result = float(value)
        return result if np.isfinite(result) and result != -9999 else None
    except (ValueError, TypeError):
        return None


def _sort_key(subject):
    return SITE_ORDER.index(subject["site"]), str(subject["id"]), subject["file_id"]


def subjects_from_phenotype(text: str, data_dir, max_per_site=None):
    """Build a deterministic candidate list without I/O (also useful for auditing).

    QC policy matches the public ABIDE PCP exclusion convention: reject any
    explicit FAIL in the five raters, missing derivatives, or unknown diagnosis.
    Missing/uncertain ratings are retained and recorded, never called a pass.
    """
    if max_per_site is not None and (not isinstance(max_per_site, int) or max_per_site < 1):
        raise ValueError("max_per_site must be a positive integer or None")
    root = Path(data_dir).resolve()
    candidates, excluded, seen = [], Counter(), set()
    for row in csv.DictReader(io.StringIO(text.lstrip("\ufeff"))):
        source_site = str(row.get("SITE_ID", "")).strip().upper()
        if source_site not in SITE_CODES:
            continue
        file_id = str(row.get("FILE_ID", "")).strip()
        ratings = {field: str(row.get(field, "")).strip() for field in QC_FIELDS}
        if any(value.upper() in {"FAIL", "FAILED"} for value in ratings.values()):
            excluded["quality_fail"] += 1
            continue
        if not file_id or file_id.lower() in {"no_filename", "nan", "none"}:
            excluded["missing_derivative"] += 1
            continue
        if Path(file_id).name != file_id or any(char in file_id for char in ("/", "\\", ":")):
            excluded["invalid_file_id"] += 1
            continue
        try:
            diagnosis = int(float(row.get("DX_GROUP", "")))
        except (ValueError, TypeError):
            diagnosis = 0
        if diagnosis not in (1, 2):
            excluded["unknown_label"] += 1
            continue
        subject_id = str(row.get("SUB_ID", file_id)).strip()
        if subject_id in seen:
            excluded["duplicate_subject"] += 1
            continue
        seen.add(subject_id)
        candidates.append({
            "id": subject_id, "file_id": file_id, "site": SITE_CODES[source_site],
            "source_site": source_site, "label": int(diagnosis == 1),
            "path": str(root / "rois_aal" / f"{file_id}_rois_aal.1D"),
            "age": _optional_float(row.get("AGE_AT_SCAN")),
            "quality": {"policy": "exclude_any_explicit_FAIL_in_five_ABIDE_raters",
                        "ratings": ratings,
                        "all_raters_ok": all(value.upper() == "OK" for value in ratings.values())},
        })
    candidates.sort(key=_sort_key)
    available = {site: sum(subject["site"] == site for subject in candidates) for site in SITE_ORDER}
    if max_per_site is not None:
        counts, selected = Counter(), []
        for subject in candidates:
            if counts[subject["site"]] < max_per_site:
                selected.append(subject)
                counts[subject["site"]] += 1
        candidates = selected
    return candidates, {"available_quality_filtered": available, "excluded": dict(excluded)}


def read_timeseries(path) -> np.ndarray:
    """Read and validate a real AAL116 time series; never substitute fake data."""
    path = Path(path)
    array = np.loadtxt(path, comments="#", dtype=np.float64, ndmin=2)
    if array.ndim != 2 or array.shape[1] != N_ROIS or array.shape[0] < 32:
        raise ValueError(f"Invalid AAL116 time series {path}: shape={array.shape}; expected T>=32,116")
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite time series: {path}")
    if not np.any(np.std(array, axis=0) > 0):
        raise ValueError(f"All ROIs constant: {path}")
    return array


def _download_subject(subject):
    path = Path(subject["path"])
    try:
        cached = path.is_file()
        if cached:
            try:
                array = read_timeseries(path)
            except (ValueError, OSError):
                cached = False
        if not cached:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".part")
            with _session().get(TIMESERIES_URL.format(file_id=subject["file_id"]), timeout=(15, 90), stream=True) as response:
                response.raise_for_status()
                with temp.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=128 * 1024):
                        if chunk:
                            stream.write(chunk)
            array = read_timeseries(temp)
            os.replace(temp, path)
        result = dict(subject)
        result.update({"n_timepoints": int(array.shape[0]), "n_rois": int(array.shape[1]),
                       "constant_rois": int(np.count_nonzero(np.std(array, axis=0) == 0)),
                       "bytes": path.stat().st_size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return result, {"status": "cached" if cached else "downloaded"}
    except Exception as exc:
        return None, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def ensure_data(data_dir, max_per_site=None, workers=6, progress: Callable | None = None) -> dict:
    """Download all selected real subjects, validating cached and new files.

    A deterministic candidate manifest is persisted *before* concurrent fetching;
    failures remain explicit and successful subjects are sorted independently of
    completion order. ``max_per_site`` is an explicit smoke-test approximation.
    Callback events are dictionaries. Requests have finite timeouts and retries.
    """
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    root = Path(data_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    phenotype = root / "Phenotypic_V1_0b_preprocessed1.csv"
    if not phenotype.is_file():
        _emit(progress, stage="phenotype", status="downloading", url=PHENOTYPE_URL)
        response = _session().get(PHENOTYPE_URL, timeout=(15, 90))
        response.raise_for_status()
        text = response.content.decode("utf-8-sig")
        if "FILE_ID" not in text.partition("\n")[0] or "DX_GROUP" not in text.partition("\n")[0]:
            raise ValueError("Phenotype endpoint did not return the expected CSV")
        temp = phenotype.with_suffix(".csv.part")
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, phenotype)
    text = phenotype.read_text(encoding="utf-8-sig")
    candidates, audit = subjects_from_phenotype(text, root, max_per_site)
    if not candidates:
        raise ValueError("No valid six-site subjects in the phenotype table")
    manifest = {
        "version": _MANIFEST_VERSION, "status": "downloading", "subjects": [], "candidates": candidates,
        "failures": [], "phenotype_url": PHENOTYPE_URL, "timeseries_url_template": TIMESERIES_URL,
        "pipeline": "cpac", "strategy": "filt_noglobal", "atlas": "aal116",
        "selection": {"site_codes": SITE_CODES, "max_per_site": max_per_site,
                      "sort": "site_order,subject_id,file_id", **audit},
        "paper_subject_counts": PAPER_COUNTS,
        "cohort_note": "Public quality-filtered availability; paper subject IDs/QC thresholds were not published. Counts are not forced to match.",
    }
    manifest_path = root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    _emit(progress, stage="manifest", completed=0, total=len(candidates), counts=audit["available_quality_filtered"])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_subject, subject): subject for subject in candidates}
        for completed, future in enumerate(as_completed(futures), 1):
            subject = futures[future]
            result, outcome = future.result()
            if result is not None:
                manifest["subjects"].append(result)
            else:
                manifest["failures"].append({"id": subject["id"], "site": subject["site"], "file_id": subject["file_id"], **outcome})
            manifest["subjects"].sort(key=_sort_key)
            manifest["failures"].sort(key=lambda item: (SITE_ORDER.index(item["site"]), item["id"]))
            if completed % workers == 0 or completed == len(candidates):
                _atomic_json(manifest_path, manifest)
            _emit(progress, stage="download", completed=completed, total=len(candidates), site=subject["site"], id=subject["id"], **outcome)
    manifest["status"] = "complete" if not manifest["failures"] else "complete_with_failures"
    manifest["counts"] = {site: sum(s["site"] == site for s in manifest["subjects"]) for site in SITE_ORDER}
    manifest["label_counts"] = {site: {str(label): sum(s["site"] == site and s["label"] == label for s in manifest["subjects"]) for label in (0, 1)} for site in SITE_ORDER}
    _atomic_json(manifest_path, manifest)
    _emit(progress, stage="complete", completed=len(candidates), total=len(candidates), counts=manifest["counts"], failures=len(manifest["failures"]))
    return manifest


def load_subjects(data_dir, max_per_site=None) -> list[dict]:
    """Load only validated downloaded subjects from a persisted manifest (offline)."""
    if max_per_site is not None and (not isinstance(max_per_site, int) or max_per_site < 1):
        raise ValueError("max_per_site must be a positive integer or None")
    root = Path(data_dir).resolve()
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"No manifest at {path}; call ensure_data first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    subjects, counts = [], Counter()
    for stored in sorted(manifest.get("subjects", []), key=_sort_key):
        subject = dict(stored)
        source = Path(subject["path"])
        if not source.is_absolute():
            source = root / source
        if not source.is_file():
            raise FileNotFoundError(f"Manifest subject {subject['id']} has no data: {source}")
        subject["path"] = str(source)
        if max_per_site is None or counts[subject["site"]] < max_per_site:
            subjects.append(subject)
            counts[subject["site"]] += 1
    return subjects


def create_subject_splits(subjects: Sequence[Mapping], seed=42, test_size=.2, fold=None, n_splits=5) -> dict:
    """Deterministically split subjects within each site, never their windows.

    Default is a stratified per-site holdout. Explicit ``fold`` selects a
    zero-based StratifiedKFold. Tiny synthetic cohorts use deterministic
    unstratified holdout, but CV refuses infeasible class counts.
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split
    ids = [str(subject["id"]) for subject in subjects]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate subject IDs: cannot guarantee no subject leakage")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one")
    if fold is not None and (not isinstance(fold, int) or not 0 <= fold < n_splits):
        raise ValueError("fold must be a zero-based index smaller than n_splits")
    result = {"train": [], "test": [], "by_site": {}, "seed": int(seed), "fold": fold, "n_splits": n_splits}
    sites = sorted({subject["site"] for subject in subjects}, key=lambda s: SITE_ORDER.index(s) if s in SITE_ORDER else len(SITE_ORDER))
    for site in sites:
        cohort = sorted((dict(s) for s in subjects if s["site"] == site), key=lambda s: str(s["id"]))
        labels = np.asarray([int(s["label"]) for s in cohort])
        index = np.arange(len(cohort))
        if len(cohort) < 2:
            raise ValueError(f"Site {site} needs at least two subjects")
        if fold is not None:
            if len(np.unique(labels)) < 2 or min(Counter(labels).values()) < n_splits:
                raise ValueError(f"Site {site} has too few subjects per class for {n_splits}-fold CV")
            train_i, test_i = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(index, labels))[fold]
        else:
            n_test = int(np.ceil(len(cohort) * test_size))
            stratify = labels if len(np.unique(labels)) == 2 and min(Counter(labels).values()) >= 2 and n_test >= 2 and len(cohort) - n_test >= 2 else None
            train_i, test_i = train_test_split(index, test_size=test_size, random_state=seed, stratify=stratify)
        site_split = {"train": [cohort[i] for i in sorted(train_i)], "test": [cohort[i] for i in sorted(test_i)]}
        result["by_site"][site] = site_split
        result["train"].extend(site_split["train"])
        result["test"].extend(site_split["test"])
    if {s["id"] for s in result["train"]} & {s["id"] for s in result["test"]}:
        raise AssertionError("Internal subject leakage")
    return result


def correlation_matrix(timeseries: np.ndarray, fisher=True, clip=1e-7) -> np.ndarray:
    """Pearson FC, optionally clipped Fisher-Z; zero diagonal and constant ROIs."""
    values = np.asarray(timeseries, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != N_ROIS or values.shape[0] < 2:
        raise ValueError("Expected time series [T>=2,116]")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite input to FC")
    if not 0 < clip < 1:
        raise ValueError("clip must lie between zero and one")
    centered = values - values.mean(axis=0, keepdims=True)
    norms = np.sqrt(np.sum(centered * centered, axis=0))
    denominator = np.outer(norms, norms)
    corr = np.divide(centered.T @ centered, denominator, out=np.zeros((N_ROIS, N_ROIS)), where=denominator > 0)
    corr = np.clip(corr, -1 + clip, 1 - clip)
    if fisher:
        corr = np.arctanh(corr)
    np.fill_diagonal(corr, 0)
    return ((corr + corr.T) * .5).astype(np.float32)


def get_fc(subject) -> np.ndarray:
    """Whole-subject Fisher-Z FC [116,116] for visualization/static controls."""
    path = subject["path"] if isinstance(subject, Mapping) else subject
    return correlation_matrix(read_timeseries(path))


def features_for_subject(subject, window=32, stride=1, max_windows=None) -> np.ndarray:
    """Return Fisher-Z upper triangles [n_windows,6670], safely cached on disk.

    The cache key includes source size/mtime, preprocessing version and window
    parameters. Even a cache hit validates the original time series first.
    ``max_windows`` deterministically subsamples evenly across the subject and
    never changes the full cached representation.
    """
    if not isinstance(window, int) or window < 2 or not isinstance(stride, int) or stride < 1:
        raise ValueError("window >=2 and stride >=1 must be integers")
    if max_windows is not None and (not isinstance(max_windows, int) or max_windows < 1):
        raise ValueError("max_windows must be a positive integer or None")
    source = Path(subject["path"] if isinstance(subject, Mapping) else subject)
    series = read_timeseries(source)
    if series.shape[0] < window:
        raise ValueError("Time series shorter than requested window")
    stat = source.stat()
    data_root = source.parent.parent if source.parent.name == "rois_aal" else source.parent
    sid = str(subject.get("id", source.stem)) if isinstance(subject, Mapping) else source.stem
    sid = "".join(c for c in sid if c.isalnum() or c in "-_")
    cache = data_root / "features" / f"{sid}_w{window}_s{stride}.npy"
    metadata_path = cache.with_suffix(".json")
    metadata = {"source": str(source.resolve()), "source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "version": 1, "window": window, "stride": stride, "fisher_clip": 1e-7}
    starts = range(0, len(series) - window + 1, stride)
    features = None
    if cache.is_file() and metadata_path.is_file():
        try:
            if json.loads(metadata_path.read_text(encoding="utf-8")) == metadata:
                candidate = np.load(cache, allow_pickle=False)
                if candidate.shape == (len(starts), N_FEATURES) and candidate.dtype == np.float32 and np.isfinite(candidate).all():
                    features = candidate
        except (ValueError, OSError, json.JSONDecodeError):
            pass
    if features is None:
        features = np.empty((len(starts), N_FEATURES), dtype=np.float32)
        for index, start in enumerate(starts):
            features[index] = correlation_matrix(series[start:start + window])[_TRIANGLE]
        cache.parent.mkdir(parents=True, exist_ok=True)
        # Unique temporary name permits concurrent readers without partial arrays.
        temp = cache.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        with temp.open("wb") as stream:
            np.save(stream, features, allow_pickle=False)
        os.replace(temp, cache)
        _atomic_json(metadata_path, metadata)
    if max_windows is not None and len(features) > max_windows:
        indices = np.linspace(0, len(features) - 1, max_windows, dtype=np.int64)
        features = features[indices]
    return features
