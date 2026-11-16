"""Truthful protection inventory for the training protocol and public-data demo.

Reads JSON run records only: never loads a model, raw subject data, keys or a
checkpoint, and never starts training/encryption. Configuration, run execution
records and detailed trace checks are deliberately distinguished.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def _read(path):
    try:
        value = path.read_text(encoding='utf-8')
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def _integer(value):
    number = _number(value)
    return int(number) if number is not None and number >= 0 and number.is_integer() else None


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _shared_sketch(value):
    """Accept only complete finite actual64-coordinate recorded vectors."""
    if not isinstance(value, list) or len(value) != 64:
        return None
    numbers = [_number(item) for item in value]
    return numbers if all(item is not None for item in numbers) else None


def describe_protection(run_dir: Path | None) -> dict:
    """Return measured/configured facts, never a security certification.

    ``observed`` means a configured mechanism has a completed-round execution
    record through the inspected runner path; it is NOT cryptographic proof.
    Private preservation and arithmetic checks require an actual trace frame.
    """
    directory = Path(run_dir) if run_dir is not None else None
    result = _read(directory / 'result.json') if directory is not None else {}
    trace = _read(directory / 'federation' / 'index.json') if directory is not None else {}
    config = _mapping(result.get('config'))
    privacy = _mapping(result.get('privacy'))
    metadata = _mapping(trace.get('metadata'))
    rows = trace.get('rounds', [])
    rows = rows if isinstance(rows, list) else []
    recorded, frame, latest_path = [], {}, None
    if directory is not None:
        for row in rows:
            round_index = _integer(row.get('round')) if isinstance(row, dict) else _integer(row)
            if round_index is not None and 1 <= round_index <= 100000:
                # Ignore untrusted row.path: read only canonical local frames.
                path = directory / 'federation' / f'round-{round_index:04d}.json'
                if path.is_file():
                    recorded.append((round_index, path))
        for round_index, path in sorted(recorded, reverse=True):
            candidate = _read(path)
            if _integer(candidate.get('round')) == round_index:
                frame, latest_path = candidate, path
                break
    completed_candidates = [_integer(result.get('round')), _integer(trace.get('completed_rounds'))]
    history = result.get('history', [])
    if isinstance(history, list):
        completed_candidates += [_integer(row.get('round')) for row in history if isinstance(row, dict)]
    completed = max((value for value in completed_candidates if value is not None), default=0)
    configured_rounds = _integer(config.get('rounds'))
    if configured_rounds is None:
        configured_rounds = _integer(result.get('total_rounds'))
    if configured_rounds is None:
        configured_rounds = _integer(trace.get('total_rounds'))
    noise_scale = _number(config.get('noise_scale', privacy.get('noise_scale')))
    clip_norm = _number(config.get('clip_norm'))
    noise_configured = noise_scale is not None and noise_scale > 0
    he_configured = config.get('homomorphic', privacy.get('homomorphic', False)) is True
    method = config.get('method')
    alignment_weight = _number(config.get('alignment_weight'))
    use_align = method is not None and method not in ('fedavg', 'no_alignment') and alignment_weight is not None and alignment_weight > 0
    clients = _mapping(frame.get('clients'))
    after = _mapping(frame.get('after'))
    aggregation = _mapping(frame.get('aggregation'))
    valid_rounds = {index for index, _ in recorded}
    events = trace.get('events', [])
    events = events if isinstance(events, list) else []
    uploads = sum(isinstance(event, dict) and event.get('event') == 'client_uploaded'
                  and _integer(event.get('round')) in valid_rounds for event in events) if frame else 0
    if frame and not uploads:
        uploads = len(clients)
    preserved = after.get('private_preserved')
    if not isinstance(preserved, bool):
        preserved = None
    raw_bytes = [_integer(client.get('upload_bytes')) for client in clients.values() if isinstance(client, dict)]
    raw_bytes = [value for value in raw_bytes if value is not None]
    common_raw_bytes = raw_bytes[0] if raw_bytes and len(set(raw_bytes)) == 1 else None
    shared_count = _integer(metadata.get('shared_parameter_count'))
    private_count = _integer(metadata.get('private_parameter_count'))
    if private_count is None:
        per_site = [_integer(_mapping(value).get('private_parameter_count'))
                    for value in _mapping(metadata.get('sites')).values()]
        if per_site and all(value is not None for value in per_site) and len(set(per_site)) == 1:
            private_count = per_site[0]
    names = metadata.get('shared_parameter_names', [])
    names = [name for name in names if isinstance(name, str)] if isinstance(names, list) else []

    site_order = trace.get('site_order')
    if not isinstance(site_order, list) or not all(isinstance(site, str) for site in site_order):
        site_order = list(clients)
    site_metadata = _mapping(metadata.get('sites'))
    preservation_by_site = _mapping(after.get('private_preserved_by_site'))
    after_sites = _mapping(after.get('sites'))
    transfer_sites = []
    for site in site_order:
        client = _mapping(clients.get(site))
        site_preserved = preservation_by_site.get(site)
        if not isinstance(site_preserved, bool):
            before_hash = client.get('private_hash')
            after_hash = _mapping(after_sites.get(site)).get('private_hash')
            site_preserved = (before_hash == after_hash) if isinstance(before_hash, str) and isinstance(after_hash, str) else None
        transfer_sites.append({
            'id': site, 'uploaded_shared_sketch': _shared_sketch(client.get('shared_sketch')),
            'clean_shared_sketch': _shared_sketch(client.get('clean_shared_sketch')),
            'local_update_sketch': _shared_sketch(client.get('update_sketch')),
            'private_preserved': site_preserved,
            'train_subjects': _integer(_mapping(site_metadata.get(site)).get('train_subject_count'))})
    aggregate_sketch = _shared_sketch(aggregation.get('shared_sketch'))
    coordinate_metadata = _mapping(metadata.get('sketch')).get('coordinates', [])
    coordinates = []
    if isinstance(coordinate_metadata, list) and len(coordinate_metadata) == 64:
        for coordinate in coordinate_metadata:
            coordinate = _mapping(coordinate)
            flat = _integer(coordinate.get('flat_index'))
            parameter = coordinate.get('parameter')
            parameter_index = coordinate.get('parameter_index')
            if flat is None or not isinstance(parameter, str) or not isinstance(parameter_index, list):
                coordinates = []
                break
            indices = [_integer(value) for value in parameter_index]
            if any(value is None for value in indices):
                coordinates = []
                break
            coordinates.append({'flat_index': flat, 'parameter': parameter, 'parameter_index': indices})
    transfer = {
        'available': bool(frame and (aggregate_sketch is not None or any(site['uploaded_shared_sketch'] is not None for site in transfer_sites))),
        'round': _integer(frame.get('round')) if frame else None, 'sketch_size': 64,
        'sites': transfer_sites, 'aggregate_shared_sketch': aggregate_sketch, 'coordinates': coordinates,
        'coordinate_note': '共享模型参数的固定64个记录坐标，不是ROI、FC或PCA坐标。',
        'alignment_statistics': {'values_recorded': False, 'marginal_mean': None, 'class_means': None,
                                 'class_counts': None, 'protected_by_noise': False, 'protected_by_paillier': False},
        'note': '数值来自最新真实trace：上传sketch为裁剪/加噪后的共享权重，clean为加噪前，aggregate为实际聚合结果；均不是Paillier密文或网络抓包。缺少记录保持未知。'}

    def observation(configured):
        if not configured:
            return 'disabled'
        if not completed:
            return 'configuration_only_no_completed_round'
        return ('completed_round_with_detailed_trace_not_cryptographic_proof' if frame and aggregation
                else 'completed_round_run_record_only_not_cryptographic_proof')

    paths = {
        'result': 'result.json' if directory is not None and (directory / 'result.json').is_file() else None,
        'federation_index': 'federation/index.json' if directory is not None and (directory / 'federation' / 'index.json').is_file() else None,
        'latest_round': f'federation/{latest_path.name}' if latest_path is not None else None,
    }
    return {
        'schema_version': 1, 'available': bool(result or trace),
        'scope': 'training_protocol_inventory_and_public_data_demo_boundaries',
        'transfer': transfer,
        'run': {'id': result.get('id', trace.get('run_id', directory.name if directory else None)),
                'status': result.get('status', trace.get('status', 'unavailable')),
                'rounds_completed': completed, 'rounds_configured': configured_rounds,
                'data_subjects': _integer(result.get('data_subjects')),
                'train_subjects': _integer(_mapping(result.get('split_counts')).get('train')),
                'cohort_scope': 'actual_selected_run_not_current_health_data_filter'},
        'deployment': {
            'mode': 'single_process_simulation', 'raw_data_centralized_in_demo': True,
            'role_switch_is_authorization': False, 'individual_data_api_available': True,
            'note': '图示为联邦训练协议边界；本演示集中读取公开研究数据，个体健康数据API可浏览，并非已部署医院隔离或权限控制。'},
        'channels': {
            'raw_data': {'uploaded': False, 'scope': 'training_protocol',
                         'items': ['raw_BOLD_timeseries', 'individual_FC', 'individual_sample_labels'],
                         'note': '不属于训练上传载荷；公开数据浏览API仍提供个体数据。'},
            'shared_model': {'uploaded': True, 'scope': 'training_protocol',
                             'parameter_count': shared_count, 'names': names,
                             'raw_tensor_bytes': common_raw_bytes,
                             'note': 'uploaded是训练协议的上传内容清单，不是实际网络抓包或已送达证明；R0也可列出允许上传的共享参数。',
                             'bytes_note': '每站点共享张量载荷，非Paillier密文线长或网络协议总字节。'},
            'alignment_statistics': {
                'uploaded': bool(use_align), 'scope': 'training_protocol',
                'items': ['marginal_mean', 'class_means', 'class_counts'] if use_align else [],
                'protected_by_paillier': False, 'protected_by_noise': False,
                'method': method, 'alignment_weight': alignment_weight,
                'class_counts_unit': 'training_window_samples_not_patients',
                'note': 'uploaded为按method/alignment_weight确定的协议清单，不是网络抓包。class_counts是训练窗口样本数，不是患者人数；这些统计量不进入共享参数加噪或Paillier路径。'},
            'private_parameters': {'uploaded': False, 'parameter_count_per_site': private_count,
                                   'items': ['private_head', 'gate'], 'scope': 'training_protocol',
                                   'note': '协议清单规定私有参数不上送；实际逐轮保留检查仅由evidence中的真实trace报告。'}},
        'mechanisms': {
            'perturbation': {'configured': noise_configured, 'observed': bool(noise_configured and completed > 0),
                             'noise_scale': noise_scale, 'clip_norm': clip_norm,
                             'distribution': 'Laplace', 'scope': 'shared_model_parameters_only',
                             'formal_dp': False, 'observation_basis': observation(noise_configured)},
            'paillier': {'configured': he_configured, 'observed': bool(he_configured and completed > 0),
                         'key_bits': _integer(config.get('paillier_bits')) if he_configured else None,
                         'keys_same_process': True, 'scope': 'shared_model_parameters_only',
                         'observation_basis': observation(he_configured)},
            'differential_privacy': {'certified': False, 'epsilon': None}},
        'evidence': {
            'recorded_rounds': len(set(index for index, _ in recorded)) if frame else 0,
            'client_uploads': uploads, 'private_preserved': preserved,
            'shared_consensus_max_error': _number(after.get('shared_consensus_max_error')),
            'equal_mean_max_error': _number(aggregation.get('equal_mean_max_error')),
            'paths': paths,
            'trace_available': bool(frame),
            'mechanism_observation_is_cryptographic_proof': False},
        'notes': [
            '数据留站点描述联邦训练协议；当前应用是集中公开数据的单机研究模拟。',
            'channels.uploaded是协议内容清单，不是网络抓包或已送达事实；运行人数仅取该次result记录，不能把120人加密冒烟等同于465人全队列。',
            '共享编码器/全局头参与聚合，私有专家头和门控不进入共享state；详细保留检查仅在真实trace存在时报告。',
            '加噪参数是Laplace尺度b，不是经过隐私会计认证的epsilon；没有正式epsilon-DP保证。',
            '真实Paillier算术仅覆盖共享模型向量；对齐均值和类别计数既未Paillier加密，也未进入参数加噪路径。',
            '同进程协调器持有原始公开数据及Paillier私钥，不声称恶意服务器无法解密或生产级安全聚合。',
            '界面角色切换仅是协议视角示意，不是访问授权；未据此声明医院隔离、认证或TLS已部署。',
            '机制observed来自配置与已完成轮次的执行记录，不是独立密码学证明；缺少trace的具体检查保持未知。',
        ],
    }
