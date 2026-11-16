/* English presentation of recorded API notes. Scientific records are not modified. */
(() => {
  'use strict';
  const translations = new Map(Object.entries({
    '正文lambda_M=0.01，汇总lambda_M=0.1；实现默认0.01并记录实际配置': 'The main text uses lambda_M=0.01, while the summary uses 0.1. This implementation defaults to 0.01 and records the actual configuration.',
    '参数加噪不等于已证明epsilon-DP；Paillier为可选真实同态聚合，默认未启用。': 'Parameter noise is not a proven epsilon-DP guarantee. Real Paillier homomorphic aggregation is optional and disabled by default.',
    '稀疏优化采样与特征上限等实现选择在每次运行配置中公开。': 'Implementation choices, including sparse-optimization sampling and feature limits, are recorded for each run.',
    '仅科研与教学演示，不用于临床诊断。': 'For research and teaching only; not for clinical diagnosis.',
    '验证集选轮': 'Validation-based checkpoint selection',
    '选轮 + 论文损失归一化': 'Checkpoint selection + paper-based loss normalization',
    'B + 聚合后刷新对齐目标': 'B + refresh alignment targets after aggregation',
    'C + L2正则0.01': 'C + L2 regularization at 0.01',
    'C + 稀疏优化400步': 'C + 400 sparse-optimization steps',
    '按验证集保存全部专家/门控': 'Save all experts and gates at the validation-selected checkpoint',
    '校正分类与域对齐的相对损失尺度': 'Correct the relative scales of classification and domain-alignment losses',
    '在广播后刷新训练对齐统计': 'Refresh training alignment statistics after broadcast',
    '有界L2/稀疏收敛对照': 'Bounded L2 and sparse-convergence controls',
    '候选仅由seed42验证集选定；锁定后一次评测，不根据测试更换候选。': 'Candidates were selected using seed-42 validation data only, then locked before test evaluation. Test results did not change the candidate.',
    '原测试集已在前一轮公开，本轮属于探索性后续验证，不是全新外部验证。': 'The original test set was previously inspected. This is exploratory follow-up validation, not a new external validation.',
    '50轮内按验证集选轮，所有私有头/门控随同恢复；比较基线也使用相同选轮规则。': 'Validation selects a checkpoint within 50 rounds; all private heads and gates are restored with it. The baseline uses the same selection rule.',
    '保留所有候选与失败，不挑最佳种子；论文79%仍为独立参考。': 'All candidates and failures are retained; the best seed is not cherry-picked. The paper-reported 79% remains a separate reference.',
    'E使用400步mask；FedAvg另做相同400步mask/标准化/划分/选轮对照，三种子掩码完全相同。': 'E uses a 400-step mask. FedAvg has a matched 400-step mask, normalization, split and checkpoint-selection control; masks match across the three seeds.',
    '验证提升没有稳定传递到测试；pooled微增不等于论文口径macro提升。': 'Validation gains did not consistently transfer to the test set. A small pooled gain is not a gain in the paper-style macro-site metric.',
    '共享模型参数的固定64个记录坐标，不是ROI、FC或PCA坐标。': '64 fixed recorded shared-model parameter coordinates; these are not ROI, FC or PCA coordinates.',
    '数值来自最新真实trace：上传sketch为裁剪/加噪后的共享权重，clean为加噪前，aggregate为实际聚合结果；均不是Paillier密文或网络抓包。缺少记录保持未知。': 'Values come from the latest recorded trace: uploaded sketches are clipped/noised shared weights, clean sketches precede noise, and aggregate sketches are actual aggregation results. These are not Paillier ciphertext or packet captures. Missing records remain unknown.',
    '图示为联邦训练协议边界；本演示集中读取公开研究数据，个体健康数据API可浏览，并非已部署医院隔离或权限控制。': 'The diagram shows the federated-training message boundary. This demo centrally reads public research data and exposes individual-data browsing APIs; it is not a deployed hospital-isolation or authorization system.',
    '不属于训练上传载荷；公开数据浏览API仍提供个体数据。': 'Not included in training uploads. Individual data remain available through the public-data browsing API.',
    'uploaded是训练协议的上传内容清单，不是实际网络抓包或已送达证明；R0也可列出允许上传的共享参数。': 'uploaded describes the protocol payload, not a packet capture or delivery receipt. Round 0 may also list permitted shared parameters.',
    '每站点共享张量载荷，非Paillier密文线长或网络协议总字节。': 'Shared tensor payload per site, not the Paillier ciphertext wire size or total network traffic.',
    'uploaded为按method/alignment_weight确定的协议清单，不是网络抓包。class_counts是训练窗口样本数，不是患者人数；这些统计量不进入共享参数加噪或Paillier路径。': 'uploaded is a protocol inventory determined by method/alignment_weight, not a packet capture. class_counts counts training windows, not patients. These statistics bypass shared-parameter noise and Paillier protection.',
    '协议清单规定私有参数不上送；实际逐轮保留检查仅由evidence中的真实trace报告。': 'Private parameters are excluded from protocol uploads. Per-round preservation checks are reported only when actual traces exist in evidence.',
    '数据留站点描述联邦训练协议；当前应用是集中公开数据的单机研究模拟。': 'Keeping data at each site describes the federated-training protocol. This application is a single-process research simulation with centrally available public data.',
    'channels.uploaded是协议内容清单，不是网络抓包或已送达事实；运行人数仅取该次result记录，不能把120人加密冒烟等同于465人全队列。': 'channels.uploaded is a protocol inventory, not a packet capture or delivery confirmation. Participant counts come from the selected result; the 120-subject encryption smoke test is not the full 465-subject cohort.',
    '共享编码器/全局头参与聚合，私有专家头和门控不进入共享state；详细保留检查仅在真实trace存在时报告。': 'The shared encoder/global head are aggregated. Private expert heads and gates are excluded from shared state; detailed preservation checks require recorded traces.',
    '加噪参数是Laplace尺度b，不是经过隐私会计认证的epsilon；没有正式epsilon-DP保证。': 'The noise parameter is Laplace scale b, not an epsilon certified by privacy accounting. No formal epsilon-DP guarantee is provided.',
    '真实Paillier算术仅覆盖共享模型向量；对齐均值和类别计数既未Paillier加密，也未进入参数加噪路径。': 'Real Paillier arithmetic covers only the shared-model vector. Alignment means and class counts are neither Paillier-encrypted nor included in parameter-noise protection.',
    '同进程协调器持有原始公开数据及Paillier私钥，不声称恶意服务器无法解密或生产级安全聚合。': 'The same-process coordinator holds the public source data and Paillier private key. This does not establish protection against a malicious server or production-grade secure aggregation.',
    '界面角色切换仅是协议视角示意，不是访问授权；未据此声明医院隔离、认证或TLS已部署。': 'View switching illustrates protocol roles; it is not authorization and does not establish hospital isolation, authentication or TLS deployment.',
    '机制observed来自配置与已完成轮次的执行记录，不是独立密码学证明；缺少trace的具体检查保持未知。': 'observed is based on configuration and completed-round execution records, not independent cryptographic proof. Checks without traces remain unknown.',
    '研究对照标签HC不等于临床健康认证；未使用训练模型或风险评分。': 'HC is a research control label, not clinical certification of health. No trained model or risk score is used.',
    '描述性群体差异，未校正混杂，不作疾病标志': 'Descriptive group differences; unadjusted for confounders and not disease markers.',
    '文件列顺序，网络示意布局非解剖坐标': 'ROI numbers follow file-column order. The network layout is illustrative, not anatomical.',
    '公开数据示例；不是训练上传内容，也不表示该示例个体参与了所选训练运行。': 'Public-data examples, not training uploads; an example does not imply participation in the selected training run.',
    '全队列HC描述性参照；HC个体已排除自身，不是规范化临床范围或患者风险。': 'Descriptive full-cohort HC reference, excluding the current HC subject. This is not a clinical normative range or a patient risk score.',
    '仅研究数据展示；HC/ASD为原始研究标签，不是当前临床诊断。': 'Research-data visualization only. HC/ASD are source research labels, not a current clinical diagnosis.',
    '此历史运行没有记录逐客户端状态，不能补造联邦回放。请选择有过程记录的实验。': 'This historical run has no recorded per-client states. No replay is fabricated; select a run with a recorded trace.',
    '真实ABIDE受试者。FC为全时序Pearson相关；预测使用32帧滑窗Fisher-z特征。AAL编号按文件列顺序，网络坐标是示意布局，不是解剖定位。仅科研演示，不用于临床诊断。': 'An actual ABIDE subject. FC uses full-series Pearson correlation; prediction uses 32-frame sliding-window Fisher-z features. AAL numbers follow file-column order. Network positions are illustrative, not anatomical. Research only; not for clinical diagnosis.',
    '公网演示只读，不能启动或停止训练': 'This public demo is read-only. Starting or stopping training is not permitted.',
    '公网演示已到期': 'This public demo has expired.',
    '未开放此公网演示路径': 'This route is not available in the public demo.',
    '演示上游暂不可用': 'The demo service is temporarily unavailable.',
    '最小年龄不能大于最大年龄': 'Minimum age must not exceed maximum age.',
    '被试标识格式无效': 'Invalid subject identifier.',
    '公开数据中没有此被试': 'This subject is not present in the public dataset.',
    '尚未开始验证集筛选，不展示模拟收益。': 'Validation-based selection has not started. No simulated gains are shown.'
  }));
  const han = /[\u3400-\u9fff]/;
  function text(value) {
    if (translations.has(value)) return translations.get(value);
    if (!han.test(value)) return value;
    let match = value.match(/^第(\d+)轮 · 共享模型已广播，私有专家保留在本站点$/);
    if (match) return `Round ${match[1]}: shared model broadcast; private experts retained locally`;
    match = value.match(/^选定特征转换\s+(\d+)\/(\d+)$/);
    if (match) return `Transforming selected features ${match[1]}/${match[2]}`;
    // Keep an inventory for verification; never silently expose a mixed-language error.
    window.EnglishUI.untranslated.add(value);
    return value;
  }
  function translate(value) {
    if (typeof value === 'string') return text(value);
    if (Array.isArray(value)) return value.map(translate);
    if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key,item]) => [key,translate(item)]));
    return value;
  }
  function message(value, fallback = 'The request could not be completed. Please try again.') {
    const result = typeof value === 'string' ? text(value) : JSON.stringify(translate(value));
    return typeof result === 'string' && !han.test(result) ? result : fallback;
  }
  window.EnglishUI = { translate, message, untranslated: new Set() };
})();
