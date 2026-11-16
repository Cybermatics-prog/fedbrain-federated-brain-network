"""Merge independently matched controls and explain the preregistered null result.
No model training, test re-evaluation, or candidate reselection occurs here.
"""
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from fedbrain.runner import atomic_json,read_json
from optimize import RESULT,report


def main():
    result=read_json(RESULT)
    if result.get('status')!='completed':
        raise RuntimeError('Wait for all locked confirmations')
    control=read_json(ROOT/'artifacts'/'optimized_fedavg_matched.json')
    assert control['selected_variant']==result['selected_variant']
    assert [r['seed'] for r in control['records']]==[42,43,44]
    checks=[]
    keys=('seed','rounds','learning_rate','local_epochs','batch_size','hidden_dim','noise_scale',
          'max_windows','max_features','sparse_max_iter','sparse_windows_per_subject','lambda_aug',
          'coupling','clip_norm','fold','test_size','validation_fraction','weight_decay','checkpoint_selection')
    for pair,base in zip(result['pairs'],control['records']):
        assert pair['seed']==base['seed']
        target=ROOT/'runs'/pair['optimized_id']
        source=ROOT/'runs'/base['id']
        a,b=read_json(target/'result.json'),read_json(source/'result.json')
        for key in keys:
            assert a['config'][key]==b['config'][key],(pair['seed'],key)
        assert read_json(target/'split.json')==read_json(source/'split.json')
        pa,pb=np.load(target/'preprocessing.npz'),np.load(source/'preprocessing.npz')
        assert np.array_equal(pa['indices'],pb['indices'])
        np.testing.assert_allclose(pa['mean'],pb['mean'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(pa['std'],pb['std'],rtol=0,atol=1e-12)
        checks.append({'seed':pair['seed'],'identical_mask':True,'identical_train_scaling':True,
                       'identical_subject_split':True,'n_features':len(pa['indices'])})
    comp=result['comparison']
    if not result.get('matched_control_audited'):
        result['fedavg_original_mask']=result['fedavg']
        comp['fedavg_original_mask_best_val_macro_mean']=comp['fedavg_best_val_macro_mean']
    result['fedavg']=control['records']
    comp['fedavg_best_val_macro_mean']=float(np.mean([r['metrics']['site_mean_accuracy'] for r in control['records']]))
    comp['fedavg_matched_mask']=True
    comp['delta_vs_matched_fedavg_macro']=comp['optimized_macro_mean']-comp['fedavg_best_val_macro_mean']
    original=[read_json(ROOT/'runs'/f'full-fedbrain-s{s}'/'result.json')['metrics'] for s in (42,43,44)]
    comp['original_pooled_mean']=float(np.mean([m['accuracy'] for m in original]))
    comp['delta_original_macro']=comp['optimized_macro_mean']-comp['original_macro_mean']
    comp['delta_original_pooled']=comp['optimized_pooled_mean']-comp['original_pooled_mean']
    result['matched_control_audited']=True
    result['matched_control_checks']=checks
    result['outcome']='passed_minimum_effect' if comp['pass_gate'] else 'no_stable_improvement'
    result['deployment']='validation_locked_candidate' if comp['pass_gate'] else 'keep_original_default'
    result['interpretation']='验证提升没有稳定传递到测试；pooled微增不等于论文口径macro提升。'
    extra='E使用400步mask；FedAvg另做相同400步mask/标准化/划分/选轮对照，三种子掩码完全相同。'
    if extra not in result['notes']:
        result['notes'].append(extra)
    atomic_json(RESULT,result)
    report(result)
    path=ROOT/'docs'/'OPTIMIZATION_RESULTS.md'
    text=path.read_text(encoding='utf-8')
    table=['','## 原始基线不能被藏掉','','|同三个种子的均值|站点macro|pooled受试者|','|---|---:|---:|',
           f"|原始固定50轮|{comp['original_macro_mean']*100:.2f}%|{comp['original_pooled_mean']*100:.2f}%|",
           f"|A：仅验证选轮|{comp['baseline_macro_mean']*100:.2f}%|{comp['baseline_pooled_mean']*100:.2f}%|",
           f"|E：验证选定候选|{comp['optimized_macro_mean']*100:.2f}%|{comp['optimized_pooled_mean']*100:.2f}%|",'',
           f"候选相对原始macro变化 **{comp['delta_original_macro']*100:+.2f}个百分点**；pooled变化{comp['delta_original_pooled']*100:+.2f}个百分点，不能只选择更好看的口径。",'',
           '## 失败验尸与停止决定','',
           f"1. 失败发生于泛化：seed42验证+{result['validation_gain_over_A']*100:.2f}pp，但三种子测试对A平均仅+{comp['delta_macro']*100:.2f}pp，低于开跑前的2pp门槛；seed44为负，原始macro也更高。",
           '2. 已排除训练不可学和明显管道错误：既有真实train-only8人功率门通过；完整私有/门控恢复、损失梯度和当前广播表示时序均有单测；独立审计另核对原始预测/受试者互斥/checkpoint重放。',
           '3. 剩余机制：400步仍未解决Eq11的联合非光滑优化，且77个验证者、最小站点6人让选轮/五候选选择波动明显；一个KKI验证分类改变即可改变macro约2.78pp。','',
           '本轮有界五候选已跑完，不再打开B/C/D测试集或根据E测试失败改选D。保留原始界面默认模型，不宣布准确率优化成功。','',
           '## 下轮优先级（未执行，不冒充有效）','',
           '1. 首先用联合近端/原始-对偶/ADMM求解Eq11，必须先通过解析反例及KKT/对偶残差，再测泛化。',
           '2. 在嵌套验证中选择lambda_aug与特征数量，而不是固定未收敛系数的top512；外层固定重复交叉验证只做最终比较。',
           '3. 若前两项仍无收益，再单独预注册多尺度动态FC+静态FC融合；不先扩大网络或噪声。','',
           '## 对照匹配审计','',
           f"400步同mask/noise/选轮FedAvg均值{comp['fedavg_best_val_macro_mean']*100:.2f}%macro；候选相对其{comp['delta_vs_matched_fedavg_macro']*100:+.2f}pp，不支持优于FedAvg。",'',
           '|seed|indices完全相同|训练scaler相同|受试者划分相同|','|---|---|---|---|']
    for check in checks:
        table.append(f"|{check['seed']}|是|是|是|")
    table+=['',f"原40步mask+best-val FedAvg为{comp['fedavg_original_mask_best_val_macro_mean']*100:.2f}%macro，作为不同预处理的辅助对照另存，不冒充同mask。",'']
    path.write_text(text+'\n'.join(table),encoding='utf-8')
    atomic_json(ROOT/'artifacts'/'optimization_canonical_numbers.json',
                {'source':'optimization_results.json, locked paired result.json, matched control result.json',
                 'selection_locked_at':result['selection_locked_at'],'comparison':comp,
                 'matched_control_checks':checks,'outcome':result['outcome']})
    print('No training or retesting. Canonical result:',comp)


if __name__=='__main__':
    main()
