"""Scoped literal translation of the recorded-federation view; no data changes."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parent.parent
pairs={
'本地学习':'Local Training','数据与私有头留在客户端':'Data and private heads stay at each client',
'上传共享参数':'Upload Shared Parameters','上传裁剪 / 加噪后的共享权重':'Upload clipped / noised shared weights',
'服务器聚合':'Server Aggregation','从真实上传权重计算聚合结果':'Aggregate the recorded uploaded weights',
'广播同步':'Broadcast','更新共享块，保留站点私有块':'Synchronize shared blocks; retain private blocks',
'轮次完成':'Round Complete','共享权重同步，私有权重独立':'Shared weights synchronized; private weights remain independent',
'该历史运行没有记录联邦过程；不根据旧指标重建轨迹。':'No federation trace was recorded for this run. Historical metrics are not used to fabricate a trace.',
'联邦记录读取失败：':'Unable to read federation records: ',
'一次联邦协作，究竟发生了什么？':'Recorded Federation Process',
'等待过程记录':'Waiting for recorded events','客户端真实参数更新':'Recorded client parameter updates',
'播放真实记录':'Play Recorded Events','下一事件 →':'Next Event →','通信轮次 ':'Communication Round ',
'仅回放真实事件，不模拟训练':'Replay recorded events only; no simulated training',
'共享编码器 / 全局专家 · 参与传输':'Shared encoder / global expert · transmitted',
'私有专家 / 门控 · 留在本地':'Private experts / gates · retained locally',
'单机六客户端模拟 · 事件高亮 ':'Single-process, six-client simulation · event highlights ',
'轮次实测快照 · 非插值动画':'recorded round snapshots · no interpolated data',
'交换共享参数与对齐统计；原始时序、私有专家 / 门控留在本地。下方 ':'Shared parameters and alignment statistics are exchanged; raw time series, private experts and gates stay local. The following ',
'为观测诊断，不是协议上传的协方差。':'is an observational diagnostic, not an uploaded covariance payload.',
'从六份本地更新，到一个共享模型':'Shared Model Parameters',
'参数坐标 · 非 ':'Parameter coordinates · not ',
'聚合前后，站点表征如何变化？':'Site Representations Before and After Broadcast',
'同一坐标系':'Fixed coordinate system',
'固定训练探针的均值与协方差；椭圆为 ':'Means and covariances of fixed training probes; ellipses show ',
'标准差描述，不是个体散点、解剖位置或置信区间。局部训练后 → 广播后变化不等于孤立 ':' standard deviation, not individual subjects, anatomical locations or confidence intervals. Local-to-broadcast changes do not isolate the causal effect of ',
'的因果收益。':'.',
'第 ':'Round ','轮记录暂不可读：':' record unavailable: ',
'已记录 · 可回放':'Recorded · replay available','真实训练记录中':'Recording training events',
'此运行暂无过程轨迹':'No trace available for this run','等待选择运行':'Select a run',
'该运行未记录联邦过程；不会根据历史准确率或损失重建协作轨迹。':'No federation trace was recorded. Accuracy or loss history is not used to reconstruct collaboration events.',
'正在查询联邦过程记录；没有记录时不生成示意数值。':'Loading federation records. Missing records are not replaced with generated values.',
'选择带联邦过程记录的运行，查看真实协作变化。':'Select a run with recorded federation events.',
'暂停回放':'Pause Replay','时间未记录':'Time not recorded',
'六客户端上传共享参数至服务器再广播的真实联邦过程':'Recorded six-client shared-parameter upload, aggregation and broadcast',
'查看 ':'Inspect ','的共享与私有更新':' shared and private updates',
'本地客户端':'Local client','训练 ':'Training: ','人 · ':' subjects · ','人':' subjects',
'窗口':' windows','共享参数 ':'Shared ', '私有块 留存':'Private · retained',
'本轮真实本地更新 ':' recorded local update ',
'中央聚合器':'Aggregator','接收共享块，私有头不上传':'Shared blocks only; private heads stay local',
'初始化就绪':'Initialization complete','等待真实记录':'Waiting for recorded data',
'不接收原始 ':'No raw ','私有头':'private heads',
'方向高亮依据真实事件；节点位置仅为协作结构示意':'Direction highlights follow recorded events; node positions illustrate collaboration, not geography',
'协作结构示意 · 尚未加载任何过程数值':'Collaboration diagram · no recorded values loaded',
'点击任一站点':'select a site','本轮已记录快照':'recorded round snapshot',
'被试 / 窗口':'subjects / windows','本地更新 ':'Local update ',
'共享参数载荷（加密前）':'Shared payload before encryption',
'裁剪 ':'Clipping ','加噪扰动 ':'noise perturbation ',
'实际裁剪因子':'Clipping factor','共享 / 私有参数数':'Shared / private parameters',
'广播保留私有头 / 门控':'Private heads / gates preserved after broadcast',
'已验证不变':'Verified unchanged','存在变化 · 请核查':'Changed · review required','尚未记录':'Not recorded',
'私有块可在本地训练中学习；这里验证的是「聚合 / 广播前后」是否保持不变，而非整轮不更新。字节数不含对齐统计、可选密文封装与协议开销。':'Private blocks may learn during local training. This check compares their state before and after aggregation/broadcast, not across the entire round. Byte counts exclude alignment statistics, optional ciphertext packaging and protocol overhead.',
'聚合器':'Aggregator','等待真实上传参数的共同坐标采样；不以 ':'Waiting for matching recorded uploaded-parameter coordinates; ',
'连接替代模型参数。':' connections are not substituted for model parameters.',
'六个客户端实际上传共享参数与聚合器参数的相同':'Matching recorded shared-parameter coordinates for six clients and the aggregator: ',
'坐标热图':'-coordinate heatmap','参数采样 ':'Parameter sample ',
'未记录':'Not recorded','共享模型参数':'Shared model parameters',
'同一组固定参数坐标 · ':'Fixed matching parameter coordinates · ',
'维采样（不是完整权重矩阵）':' sampled dimensions, not the full weight matrix',
'统一色阶 ':'Shared scale ', '服务器共享更新 ':'Server shared update ',
'上传权重到聚合均值距离':'Uploaded-weight distance to aggregate mean',
'聚合算术最大误差':'Maximum aggregation arithmetic error','广播后共享最大差':'Maximum shared-state difference after broadcast',
'等待固定训练探针的真实投影统计；不绘制模拟站点分布。':'Waiting for recorded fixed-training-probe projections. Site distributions are not simulated.',
'同一固定训练':'Fixed training-only ',
'坐标下，本地训练后和广播后的站点均值及协方差椭圆':' coordinates: site means and covariance ellipses after local training and broadcast',
'广播同步后':'After Broadcast','本地训练后 · 聚合前':'After Local Training · Before Aggregation',
'同范围':'same scale','训练探针窗口 · ':'training-probe windows · ',
'类中心 · ':'class centroids · ','类中心':'class centroids','站点均值 · ':'site mean · ',
'固定于初始训练探针':'fixed to initial training probes','解释方差 ':'explained variance '
}
path=ROOT/'web'/'federation.js'
source=path.read_text(encoding='utf-8')
for before,after in sorted(pairs.items(),key=lambda item:-len(item[0])):
    source=source.replace(before,after)
source=source.replace("'zh-CN'","'en-US'")
source=source.replace('return await r.json();','const payload=await r.json();return window.EnglishUI?.translate(payload)??payload;')
remaining=re.findall(r'[\u3400-\u9fff]+',source)
if remaining: raise RuntimeError('Untranslated federation fragments: '+repr(remaining))
path.write_text(source,encoding='utf-8')
print(json.dumps({'federation_replacements':len(pairs),'remaining_cjk':len(remaining)}))
