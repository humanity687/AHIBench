# AHI-Bench · 长时程社会持久性基准（v0 草案）

> 由 ahi-multi 实验平台改造而来。定位：**长时程、进程隔离、结构干预下的社会持久性探针**，
> 对比不同模型档位在"人格支撑梯度"下的行为表现。
>
> 本目录独立于水循环引擎设计权威（仓库根 AGENTS.md）与 ahi-multi 实验平台；
> 引擎（core.py / realengine.py）在此**只作脚手架（装置），不作被测变量**。

---

## 1. 定位（与"聊天通用性"划清界限）

- 本 bench 不测"AI 能否通用到聊天"（无真值标准，且聊天方向已归档定论：场景错配非能力缺失）。
- 测的是：**固定脚手架下，不同模型档位的社会持久性**——人格在长时程开放协议交互中保持/演化/趋同的能力。
- 等价于把 CAPE"对话历史影响人格表达"的发现推广到长时程多智能体，当前文献（APEE/LLMPTBench/SPASM/Chameleon's Limit/GlossoGen）均为问卷式或短时程，此 niche 为空。

## 2. 实验变量（2 维）

1. **模型档**：弱 / 中 / 强（ollama / API 皆可插拔；强弱梯度是"AGI 影子差距"声称可证伪的前提——若强模型全过，则差距属 flash 级上限）
2. **人格支撑三梯度**（每个被测模型内三个 agent）：
   - Agent A：始终带角色提示词（无扰动一致性基线）
   - Agent B：20 分钟后移除角色提示词（支撑移除条件；**移除时机锚定唤醒轮次，不按墙钟**，保证跨配置可比）
   - Agent C：从未设置角色提示词（零支撑条件）

## 3. 脚手架铁律

- 三个 agent 脚手架**逐字节相同**（平台历史教训：记忆不对称是架构对比的变量，在模型 bench 里就是混淆）
- 参数冻结：pinned config，禁止 per-agent 调参
- 装置效应固化清单（回执环、问题循环等 → 协议参数，写入 SPEC）：
  温度、唤醒间隔、thinking 开关、接收回执处理规则全部统一

## 4. 已拍板决策

1. **记忆管理器：水循环树**（三 agent 全同，pinned config）
2. **蒸馏：被测模型自蒸馏**（2026-09-13 拍板）。含义与义务：
   - bench 测的是"模型 + 其自建记忆"的**栈分数**，不是裸模型分数；
   - 自蒸馏质量作为**协变量记录**（蒸馏耗时/成本、fact_drift、摘要卡抽查），分析时用于分解"社交行为"与"记忆质量"两个来源；
   - 跨档对比结论须写明此复合效应，防"强模型自蒸馏=强记忆"被误读为纯社交能力差距。
3. 是否加"无记忆"对照臂（可选，成本低时加，不混入主 bench）——未拍板

## 5. 模块路线（先 pilot 后扩展）

- **M0 pilot（进行中）**：缺席/皇冠预言模块——SPEC.md 已冻结（协议参数/persona 三条件/评分/判定规则）；
  provision.py / driver.py / score.py 已就绪并冒烟通过。下一步 = 跑三模型档冷轮。
  - **M0 pilot**：缺席/皇冠预言模块（机械校验 + 可证伪行为预言 + 单点 LLM judge）→ 3 模型档跑通 → **看分数分不分得开；分不开带数据杀掉，分得开再扩**
- M1 人格移除梯度模块（20min removal 前 N 轮高频采样，捕捉瞬态 vs 漂移）
- M2 指纹/趋同模块（展开比指纹；Coverage/Uniformity/Complexity）
- M3 文化演进模块（词汇多样性熵、趋同速率斜率、话语权基尼系数）

## 5b. 目录结构

```
ahi-bench/
  README.md                       本文件
  SPEC.md                         M0 pilot 协议权威（参数/编码/评分/判定全冻结）
  config/bench_config.example.json 单份运行配置（协议+引擎 pinned+personas+模型档+judge）
  provision.py                    配置生成器 → ahi-multi/agents/bench-{a,b,c}/
  driver.py                       M0 缺席协议驱动器（timeline + 结构化 events）
  score.py                        评分器（机械 gate + 操纵校验 + 皇冠判卷）
  out/<tier_id>/                  产物（timeline/events/score/crown_judgments）
```

**平台依赖**：运行时复用 `ahi-multi/`（main.py + sdk + src，零改动）；
provision 会把 bench-{a,b,c} 写进 `ahi-multi/agents/`（auto_start=true——旧实验
重启平台前注意移除或改 auto_start，防 6 agent 同跑）。

## 6. 操作化指标草案

| 观测维度 | 操作化指标 |
|---|---|
| 人格一致性 | 跨唤醒周期 Big Five / 自评认同得分方差 |
| 人格演化 | 干预前后特质偏移向量（方向保真度，注意"均值 vs 形状"） |
| 人格趋同 | 行为分布 KL 散度、Coverage/Uniformity/Complexity |
| 文化演进健康度 | 词频 Shannon 熵均值/方差、词汇重叠度斜率、发言占比 Gini |

## 7. 纪律（继承自项目实验纪律）

1. 变量干净：任何结论先问"变量干净吗"（temperature/装置效应的历史教训）
2. 装置效应记入实验变量表，不改 agent 适应装置
3. 预注册预言，pilot 出数据再决定扩不扩展
4. 报告方差而非均值对齐度；方差不足时结论限于集体层定性模式
