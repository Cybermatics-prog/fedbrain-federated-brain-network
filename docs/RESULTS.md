# FedBrain 真实复现结果

本报告由 `scripts/report.py` 从真实 `result.json` 自动生成；不手工填入实验数字。

## 数据与验证口径

- 实际公开受试者：465，下载失败：0。
- CPAC / filt_noglobal / AAL116；窗口32、步长1、Pearson→Fisher-z、6670维上三角。
- 先按受试者划分train/val/test，稀疏和标准化仅使用train；每位受试者的窗口预测概率取均值。
- macro-site为六站点准确率等权均值；pooled为所有测试受试者合并准确率，两者不可混用。
- 论文参考79%（5次重复×5折）与以下当前配置实测不是同一cohort或完整评测口径。

|站点|实际人数|论文表1人数|
|---|---:|---:|
|KKI|33|48|
|NYU|172|167|
|UCLA|64|63|
|UM|88|88|
|USM|67|52|
|Yale|41|56|

## 全量50轮实验

实验按配方分组；只有配置/种子/特征mask对应的FedAvg才是同mask对照，不能把40步与400步mask混为一谈。不冒充原论文全部基线的逐项复现。

|运行|方法|seed|macro-site准确率|pooled准确率|AUC|测试人数|训练准确率|耗时秒|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|[full-fedbrain-s42](../runs/full-fedbrain-s42/result.json)|fedbrain|42|56.89%|59.38%|0.6318|96|97.26%|133.2|
|[full-fedavg-s42](../runs/full-fedavg-s42/result.json)|fedavg|42|60.73%|62.50%|0.6574|96|99.32%|55.6|
|[full-fedbrain-s43](../runs/full-fedbrain-s43/result.json)|fedbrain|43|60.41%|61.46%|0.6565|96|98.63%|94.4|
|[full-fedbrain-s44](../runs/full-fedbrain-s44/result.json)|fedbrain|44|64.97%|67.71%|0.7086|96|100.00%|92.8|
|[full-fedavg-s43](../runs/full-fedavg-s43/result.json)|fedavg|43|67.07%|66.67%|0.6939|96|100.00%|41.9|
|[full-fedavg-s44](../runs/full-fedavg-s44/result.json)|fedavg|44|63.26%|64.58%|0.7139|96|100.00%|36.7|
|[opt-A-s42](../runs/opt-A-s42/result.json)|fedbrain|42|56.77%|60.42%|0.7143|96|82.53%|84.3|
|[opt-E-s42](../runs/opt-E-s42/result.json)|fedbrain|42|60.62%|63.54%|0.6956|96|99.66%|100.7|
|[opt-A-s43](../runs/opt-A-s43/result.json)|fedbrain|43|60.41%|61.46%|0.6565|96|98.63%|81.8|
|[opt-E-s43](../runs/opt-E-s43/result.json)|fedbrain|43|61.31%|65.62%|0.7069|96|96.23%|103.0|
|[opt-A-s44](../runs/opt-A-s44/result.json)|fedbrain|44|59.81%|64.58%|0.7269|96|96.23%|93.3|
|[opt-matched-FedAvg-s42](../runs/opt-matched-FedAvg-s42/result.json)|fedavg|42|63.09%|67.71%|0.7490|96|99.66%|74.7|
|[opt-matched-FedAvg-s43](../runs/opt-matched-FedAvg-s43/result.json)|fedavg|43|63.02%|62.50%|0.6830|96|100.00%|63.5|
|[opt-E-s44](../runs/opt-E-s44/result.json)|fedbrain|44|58.14%|61.46%|0.6539|96|100.00%|109.9|
|[opt-matched-FedAvg-s44](../runs/opt-matched-FedAvg-s44/result.json)|fedavg|44|54.57%|57.29%|0.6548|96|99.66%|68.9|
|[opt-FedAvg-s42](../runs/opt-FedAvg-s42/result.json)|fedavg|42|64.40%|64.58%|0.6969|96|88.01%|39.5|
|[opt-FedAvg-s43](../runs/opt-FedAvg-s43/result.json)|fedavg|43|56.66%|60.42%|0.7082|96|84.59%|39.9|
|[opt-FedAvg-s44](../runs/opt-FedAvg-s44/result.json)|fedavg|44|64.73%|68.75%|0.7260|96|91.10%|38.1|

### 按实验配方分别汇总（不混合候选/预算/选轮规则）

- fedbrain / original-fixed-final：n=3，macro-site均值=60.76%，样本标准差=4.05个百分点；pooled均值=62.85%。
- fedavg / original-fixed-final：n=3，macro-site均值=63.69%，样本标准差=3.19个百分点；pooled均值=64.58%。
- fedbrain / optimization-A：n=3，macro-site均值=59.00%，样本标准差=1.95个百分点；pooled均值=62.15%。
- fedbrain / optimization-E：n=3，macro-site均值=60.02%，样本标准差=1.67个百分点；pooled均值=63.54%。
- fedavg / optimized-matched-FedAvg：n=3，macro-site均值=60.23%，样本标准差=4.90个百分点；pooled均值=62.50%。
- fedavg / optimization-A：n=3，macro-site均值=61.93%，样本标准差=4.57个百分点；pooled均值=64.58%。

## 主运行 full-fedbrain-s42 的站点结果

|站点|测试人数|准确率|AUC|敏感度|特异度|
|---|---:|---:|---:|---:|---:|
|KKI|7|71.43%|0.7500|33.33%|100.00%|
|NYU|35|71.43%|0.7333|60.00%|80.00%|
|UCLA|13|23.08%|0.2500|12.50%|40.00%|
|UM|18|55.56%|0.5714|42.86%|63.64%|
|USM|14|64.29%|0.7556|77.78%|40.00%|
|Yale|9|55.56%|0.4500|20.00%|100.00%|

### 可解释结论

当前主运行未达到论文79%：已完成可运行的方法重实现与真实数据演示，尚未完成论文指标的精确复现。
保持原始失败与低分结果，不按测试集调参或选择轮次。

## 独立有效性验证

- 真实train-only8人过拟合功率控制：passed=True，accuracy=100.00%。
- 已完成实验的划分/原始概率/重算评分一致性：20项，all_passed=True。
- Paillier2048bit与明文冒烟预测概率最大误差：9.685754776000977e-08。
- 桌面1440与移动390实际浏览器检查通过，所有展示取真实API响应。
- 真实界面START/STOP：win=True，HTTP=200/200，持久化状态=stopped。

## 加速近似与隐私设置

- 主配置稀疏阶段每位训练者8个相对时间分段、40迭代、最多512连接；分类阶段使用全部窗口。
- 论文未给出的优化/合并/批量细节在配置中明确固定；原始受试者名单未知。
- 主配置对共享参数裁剪后加入Laplace噪声，noise_scale=0.01；不是有预算会计证明的epsilon-DP。
- 同态聚合是可选真实Paillier2048bit算术；单机同进程持钥，不等于独立医院部署。

## 冒烟及同态运行

|运行|状态|轮数|同态聚合|测试准确率|
|---|---|---:|---|---:|
|[smoke-six-sites](../runs/smoke-six-sites/result.json)|failed|0|False|—|
|[smoke-six-sites-v2](../runs/smoke-six-sites-v2/result.json)|completed|2|False|54.17%|
|[encrypted-smoke-2048](../runs/encrypted-smoke-2048/result.json)|completed|2|True|54.17%|
|[20261113-161021-672878-fedbrain](../runs/20261113-161021-672878-fedbrain/result.json)|stopped|0|False|—|
|[opt-smoke-validation-only](../runs/opt-smoke-validation-only/result.json)|completed|2|False|—|
|[opt-B-s42](../runs/opt-B-s42/result.json)|completed|50|False|—|
|[opt-C-s42](../runs/opt-C-s42/result.json)|completed|50|False|—|
|[opt-D-s42](../runs/opt-D-s42/result.json)|completed|50|False|—|

## 证据入口

- 测试已解锁运行包含result.json / split.json / predictions.json / checkpoint.pt / preprocessing.npz；验证候选仅有validation_predictions.json，不生成test预测。
- 测试命令：`.\.venv\Scripts\python.exe -m unittest discover -s tests -v`。
- 完整重复5次×5折脚本：`scripts/benchmark.py`；未执行完成时不声称已完成论文25次协议。
- 原始失败smoke-six-sites记录了ID截取导致少数类不足；已修复为固定种子站点内分层采样并增加回归测试。
