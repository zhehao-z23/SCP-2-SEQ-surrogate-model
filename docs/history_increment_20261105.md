# Byte 增量核验：训练规模问题

2026-11-05；只核影响本轮扩大训练的原件，不重新历史盘点。

| 问题 | 原件核验 | 对本轮的影响 |
|---|---|---|
| entry_v1是否是全部实验 | 否。`deep_scan_20261013`已完成33架构族×5外折，302内层+165外层=467次，0失败；另有`night_20261013`71成功试验保留 | 不重开小样本广扫；用少量历史代表锚点检查规模效应 |
| 旧共同数据 | 1,061 U / 718组，seed1701；历史顶点等权p | 小样本与单seed仍是关键限制；不是新面积C任务 |
| 当前历史最佳 | 组成电荷MLP H=.056579；小序列Transformer .056669；ESMC600 mean+charge deep .060391；ESMC300同路线 .060504 | 保留强简单基线、从头顺序模型和pLM融合代表，不根据entry_v1淘汰全部pLM |
| 容量/调参覆盖 | 最大读出头13,723,199参数；每family/fold仅1–3trial，全部未进入4trial后的TPE阶段 | 深头非未尝试；进一步泛扫容量收益不明确，优先数据规模与优化校准 |
| 数值证据 | `results/verification_20261015`将33族原OOF复算，与汇总差≤1e-12；选型后bootstrap仅描述性 | 复用原结果，不必重新算同身份fit |
| 更大合法T开发集 | release_v1中development & static_eligible & label_join_valid=21,921行、21,915唯一T、7,999组 | 可建立单独历史T→p开发合同开展嵌套规模曲线 |
| T同源连接 | 全部21,921行的T hash在all_sequence_groups均存在，group与sample manifest一致，缺失/不一致均0 | 复用已有T/U/C关系图及protected闭包，不再同条件全量搜索 |
| 各折大小 | fold0..4=4,366 / 4,390 / 4,440 / 4,250 / 4,475行 | 仅本历史开发角色；不得冒充新C正式五折 |
| 大T缓存 | Byte已核路径的ESMC mean为1140 U入口缓存，night三encoder缓存1061 U；g1旧大缓存有索引但数值/生成身份未充分验证 | 本轮直接复用已审查的LocalHFBackend在Byte提取21,915 T mean；不移动大matrix |

旧深扫第一/第二名差0.0000902，配对group bootstrap95%区间[-.001041,.001269]，跨0且不全落预设±.001范围，既不是明确胜负也未证明实用等效。ESMC600融合相对简单基线差+.003813，描述性区间[+.002324,+.005256]。

本轮T队列与旧严格U队列输入语义不同，不能直接用跨轮分数断言规模改变了排名；应在同一T评价集上按组嵌套增加训练规模，另报严格T=U子集表现。

原件位置：

- `${LEGACY_ROOT}/deep_scan_20261013/STATUS.md`
- `${LEGACY_ROOT}/deep_scan_20261013/results/verification_20261015/REPORT.md`
- `${LEGACY_ROOT}/deep_scan_20261013/results/verification_20261015/RESULTS_VERIFICATION.json`
- `${LEGACY_ROOT}/deep_scan_20261013/results/family_comparison.csv`
- `${LEGACY_ROOT}/deep_scan_20261013/configs/deep_protocol.json`
- `${LEGACY_ROOT}/data/qualification/release_v1/sample_manifest.parquet`
- `${LEGACY_ROOT}/data/qualification/release_v1/all_sequence_groups.parquet`
- `${LEGACY_ROOT}/night_20261013/data/encoders/`
- `${LEGACY_ROOT}/night_20261013/package/src/seq2scp_training/hf_backend.py`

本次已直接读取结果报告、协议、原址Parquet和缓存身份；没有再次重算历史全体OOF。7月/5月不在本次重审范围。
