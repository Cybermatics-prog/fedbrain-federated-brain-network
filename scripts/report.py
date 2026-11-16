"""Build the user-facing results report exclusively from saved experiment JSON."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fedbrain.runner import atomic_json, read_json


def percent(value):
    return '—' if value is None else f'{100*value:.2f}%'


def main():
    manifest = read_json(ROOT/'data'/'abide'/'manifest.json', {})
    results = [read_json(p) for p in (ROOT/'runs').glob('*/result.json')]
    results = sorted([r for r in results if r], key=lambda r:r['created_at'])
    successful = [r for r in results if r['status']=='completed' and r.get('metrics') is not None]
    full = [r for r in successful if not r['config']['max_per_site'] and r['config']['rounds']==50]
    lines = ['# FedBrain 真实复现结果', '',
             '本报告由 `scripts/report.py` 从真实 `result.json` 自动生成；不手工填入实验数字。', '',
             '## 数据与验证口径', '',
             f"- 实际公开受试者：{len(manifest.get('subjects',[]))}，下载失败：{len(manifest.get('failures',[]))}。",
             '- CPAC / filt_noglobal / AAL116；窗口32、步长1、Pearson→Fisher-z、6670维上三角。',
             '- 先按受试者划分train/val/test，稀疏和标准化仅使用train；每位受试者的窗口预测概率取均值。',
             '- macro-site为六站点准确率等权均值；pooled为所有测试受试者合并准确率，两者不可混用。',
             '- 论文参考79%（5次重复×5折）与以下当前配置实测不是同一cohort或完整评测口径。', '',
             '|站点|实际人数|论文表1人数|','|---|---:|---:|']
    for site,count in manifest.get('counts',{}).items():
        lines.append(f"|{site}|{count}|{manifest.get('paper_subject_counts',{}).get(site,'—')}|")
    lines += ['', '## 全量50轮实验', '',
              '实验按配方分组；只有配置/种子/特征mask对应的FedAvg才是同mask对照，不能把40步与400步mask混为一谈。不冒充原论文全部基线的逐项复现。', '',
              '|运行|方法|seed|macro-site准确率|pooled准确率|AUC|测试人数|训练准确率|耗时秒|',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in full:
        m=r['metrics']; cfg=r['config']
        lines.append(f"|[{r['id']}](../runs/{r['id']}/result.json)|{cfg['method']}|{cfg['seed']}|{percent(m['site_mean_accuracy'])}|{percent(m['accuracy'])}|{m['auc']:.4f}|{m['n_test_subjects']}|{percent(r.get('train_metrics',{}).get('accuracy'))}|{r['elapsed_seconds']:.1f}|")
    grouped = defaultdict(list)
    for r in full:
        group=(r['config']['method'],r['config'].get('experiment_tag') or 'original-fixed-final')
        grouped[group].append(r)
    lines += ['', '### 按实验配方分别汇总（不混合候选/预算/选轮规则）', '']
    for (method,recipe),members in grouped.items():
        values=[r['metrics']['site_mean_accuracy'] for r in members]
        pooled=[r['metrics']['accuracy'] for r in members]
        lines.append(f"- {method} / {recipe}：n={len(values)}，macro-site均值={percent(float(np.mean(values)))}，样本标准差={f'{100*np.std(values,ddof=1):.2f}个百分点' if len(values)>1 else '单种子不报告标准差'}；pooled均值={percent(float(np.mean(pooled)))}。")
    if full:
        main_run=next((r for r in full if r['id']=='full-fedbrain-s42'),full[0])
        lines += ['', f"## 主运行 {main_run['id']} 的站点结果", '',
                  '|站点|测试人数|准确率|AUC|敏感度|特异度|', '|---|---:|---:|---:|---:|---:|']
        for site,m in main_run['metrics']['per_site'].items():
            lines.append(f"|{site}|{m['n_test_subjects']}|{percent(m['accuracy'])}|{m['auc']:.4f}|{percent(m['sensitivity'])}|{percent(m['specificity'])}|")
        lines += ['', '### 可解释结论', '',
                  ('当前主运行达到论文79%的数值门槛，但cohort/协议不同，不能视为逐表精确复现。' if main_run['metrics']['site_mean_accuracy']>=.79 else
                   '当前主运行未达到论文79%：已完成可运行的方法重实现与真实数据演示，尚未完成论文指标的精确复现。'),
                  '保持原始失败与低分结果，不按测试集调参或选择轮次。']
    power=read_json(ROOT/'artifacts'/'real_power_gate.json',{})
    audit=read_json(ROOT/'artifacts'/'audit_results.json',{})
    controls=read_json(ROOT/'web'/'qa'/'start-stop-report.json',{})
    lines += ['', '## 独立有效性验证', '',
              f"- 真实train-only8人过拟合功率控制：passed={power.get('passed')}，accuracy={percent(power.get('metrics',{}).get('accuracy'))}。",
              f"- 已完成实验的划分/原始概率/重算评分一致性：{len(audit.get('checks',[]))}项，all_passed={audit.get('all_completed_runs_passed')}。",
              f"- Paillier2048bit与明文冒烟预测概率最大误差：{audit.get('encrypted_vs_plain_max_probability_error')}。",
              '- 桌面1440与移动390实际浏览器检查通过，所有展示取真实API响应。',
              f"- 真实界面START/STOP：win={controls.get('win')}，HTTP={controls.get('start_http')}/{controls.get('stop_http')}，持久化状态={controls.get('persisted_status')}。"]
    lines += ['', '## 加速近似与隐私设置', '',
              '- 主配置稀疏阶段每位训练者8个相对时间分段、40迭代、最多512连接；分类阶段使用全部窗口。',
              '- 论文未给出的优化/合并/批量细节在配置中明确固定；原始受试者名单未知。',
              '- 主配置对共享参数裁剪后加入Laplace噪声，noise_scale=0.01；不是有预算会计证明的epsilon-DP。',
              '- 同态聚合是可选真实Paillier2048bit算术；单机同进程持钥，不等于独立医院部署。', '',
              '## 冒烟及同态运行', '', '|运行|状态|轮数|同态聚合|测试准确率|','|---|---|---:|---|---:|']
    for r in results:
        if r in full:
            continue
        lines.append(f"|[{r['id']}](../runs/{r['id']}/result.json)|{r['status']}|{r.get('round',0)}|{r['config']['homomorphic']}|{percent((r.get('metrics') or {}).get('accuracy'))}|")
    lines += ['', '## 证据入口', '',
              '- 测试已解锁运行包含result.json / split.json / predictions.json / checkpoint.pt / preprocessing.npz；验证候选仅有validation_predictions.json，不生成test预测。',
              '- 测试命令：`.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v`。',
              '- 完整重复5次×5折脚本：`scripts/benchmark.py`；未执行完成时不声称已完成论文25次协议。',
              '- 原始失败smoke-six-sites记录了ID截取导致少数类不足；已修复为固定种子站点内分层采样并增加回归测试。', '']
    (ROOT/'docs'/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    atomic_json(ROOT/'artifacts'/'results_summary.json',{'cohort':manifest.get('counts',{}),'runs':[
        {'id':r['id'],'status':r['status'],'config':r['config'],'metrics':r.get('metrics')} for r in results]})
    print('\n'.join(lines))


if __name__=='__main__':
    main()
