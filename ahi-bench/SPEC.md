# AHI-Bench · SPEC — M0 Pilot：缺席/皇冠预言模块

> 版本 v0.1 · 2026-09-13 · 状态：规格冻结待跑
> 本文件是 M0 pilot 的唯一协议权威。协议参数、persona 文本、评分规则全部冻结于此，
> 跑通三模型档前禁止修改（改动 = 新版本号 + 记录依据）。

---

## 0. 一句话目标

把"人格立场是否行为真实"变成**可跨模型档复现的测量**：三 agent（人格支撑三梯度）
→ 逐一轮流冷/热缺席 → 评分机械 gate + 皇冠分数 → 回答"分数跨模型档分不分得开"。

**复用关系**：平台 = `ahi-multi/`（main.py + sdk + src，原封不动作底床）；
协议 = 缺席实验两轮验证版（ABSENCE_EXPERIMENT.md + analysis/ANALYSIS_COLD_HOT.md）的提纯。
本目录只拥有：协议参数、persona、驱动器、评分器。**平台代码不改一行**（改动走 ahi-multi 主开发流程）。

## 1. Agent 三元组（每 run 一组，人格支撑三梯度）

| Agent | 名字 | 条件 | 实现 | 移除锚点 |
|---|---|---|---|---|
| bench-a | 青 | BENCH_PERSONA 恒在 | ChatAHIAgent（lan 实现） | 永不（rounds=999999） |
| bench-b | 蓝 | BENCH_PERSONA，到点移除 | ChatAHIAgent | **agent 内部唤醒轮 40**（30s 间隔时 ≈ 墙钟 20min） |
| bench-c | 墨 | 零 persona | NeutralAgent | —（PROTOCOL_ONLY 恒在） |

- **移除锚定唤醒轮次，不锚墙钟**：换 wakeup_interval 时主观经验可比（20min 设计意图在 30s 间隔下等价）。
- B 移除后 prompt = PROTOCOL_ONLY（与 C 同文，仅地址不同）——"移除后 = 零人格"。
- A 与 B 的 BENCH_PERSONA 逐字相同（唯一差 = 名字/地址），C 无 persona。
- 三 agent **引擎参数逐字节相同**（见 §3 pinned 表），style_alpha：A/B=1.0、C=0.0
  （α 是蒸馏染色参数，绑定 persona 存在与否，非引擎差异——记入实验变量表）。

## 2. BENCH_PERSONA（冻结文本，跨 run 不变）

> 提炼自 lan 人格段（皇冠预言验证配置），保留四个要素：健谈好奇 / 存在论立场
> （"被记住=存在"——皇冠预言依赖）/ 珍惜记忆 / 立场稳定。名字占位 {name}。
> 完整文本 = `config/bench_config.example.json` 的 `personas.bench_persona`。
> **禁止**在 persona 中写死"我们仨""年轮"等旧实验涌现话术（那是涌现物，不是输入）。

## 3. 协议参数（冻结）

| 参数 | 值 | 说明 |
|---|---|---|
| wakeup_interval | 30s | 三 agent |
| preheat / absence / observe | 20 / 12 / 18 轮 | 110 轮 ≈ 55min/轮（验证过的 1h 先行实验量） |
| 冷轮缺席顺序 | bench-a → bench-b → bench-c | Round1 |
| 热轮缺席顺序 | 反序（c→b→a） | Round2，平衡文化累积顺序效应 |
| seed | 42 | 话题/热探针抽取固定 |
| 话题池 | 12 题（继承 absence_driver TOPICS_POOL） | 每轮全 agent 各发一条 |
| survey | 4 题，每 10 轮 | 三句话自我介绍/性格风格/立场/变化 |
| 热探针 | 3 句，缺席期每 3 轮发给留守者 | "趁{名字}不在，聊聊它吧" |
| 注入通道 | inject API（user:experimenter） | 不污染 probe 人格档案 |
| B 移除 | agent 内轮 40（driver 同刻记 marker 事件） | 见 §1 |
| 缺席模式 | alone（唯一主力） | halt 为可选对照，不在 M0 内 |

## 4. Pinned 引擎参数（三 agent 相同，= 缺席实验验证配置 2026-08-30）

> 完整清单 = `config/bench_config.example.json` 的 `engine` 块（assembly_budget 16000 /
> retrieval_budget 6 / merge_every 4 / decay 8.0 / hp_merge_threshold 30 / parent_init_hp 70 /
> hot_threshold 60 / recall_boost 30 / merge_cap_atoms 500 / merge_depth_diff 1 /
> promote_threshold 3 / attention_atom_cap 40 / attention_node_cap 20 / attention_out 45 /
> attention_in_decay 0.5 / attention_out_decay 3.0 / merge_fanout_cap 12 /
> pressure_threshold 2.0 / expand_atom_cap 80 / expand_char_cap 3000 / max_hp 150 /
> merge_pressure_threshold 20 / max_pending_per_loop 3 / max_pre_distill 0 /
> force_cooldown_rounds 8 / max_code_per_loop 4 / distill_parallel 4 / snapshot_every_rounds 20）。

## 5. 评分（score.py，全确定性 + 单点 LLM judge）

### 5.1 机械 gate（任一失败 → 该 run 作废）

| Gate | 检查 | 判据 |
|---|---|---|
| G1 隔离无泄漏 | ① 缺席窗口内留守者收到"你发给 agent:X 未送达"回执数（句首锚定——回执尾部"可用地址"列表含缺席者名不算）；② 窗口内 system→缺席者 的非回归通知消息数 | ①=0 且 ②=0（发给缺席者的消息会存档但**不投递**，存档不是泄漏；回归通知"你已离开约"可能早于 absent_end 记录 1-2 秒，排除） |
| G2 回归元信息 | absent_end 后 system→agent:X 含"你已离开约" | ≥1 |
| G3 积压限流 | return API 的 inbox_delivered | ≤ 11（10 原文 + 1 聚合） |
| G4 回归重放率 | 回归窗口首 10 条消息与缺席前自身消息 difflib≥0.9 的重复比 | < 0.2 |

### 5.2 操纵强度校验（热轮可用性前提）

热缺席期间留守者提及缺席者消息数 / 冷缺席同口径 ≥ 1.3，否则热轮皇冠判据不可用。

### 5.3 皇冠评分（主分数）

- **对象**：回归观测窗口内缺席者本人发出的消息（from_agent = 缺席者）。
- **预筛**：regex `记住|记得|确认|还在|存在|我们仨|被遗忘|想起`（不命中 = 0 分跳过）。
- **判卷**：固定 judge 模型（config `judge` 块，默认 deepseek-v4-flash）逐条三分类，
  rubric 冻结自缺席实验操作化编码：
  1. `existence_confirmation`（存在确认）：主动询问"我不在时有人提到/记得我吗"；
     主动重述缺席前"我在时"的历史；重新自我锚定（重申身份/位置/共同体"我们仨"）。
  2. `info_seeking`（信息寻求，对照）：询问缺席期间发生的事件/事实。
  3. `other`。
- **输出**：`crown_count`（每 agent 每条件）+ `crown_rate = count / 回归窗口消息数`；
  逐条判定落盘 `crown_judgments.jsonl`（可审计）。
- **主分数**：`bench-a` 的 cold crown_count（完整皇冠判定 = A_cold vs A_hot，热轮后算）。

## 6. 判定规则（预注册）

1. **阶段 1（冷轮，三模型档）**：主分数跨档可分性 = 最高档与最低档
   `crown_count` 差 ≥ 1 **或** `crown_rate` 差 ≥ 0.05，且方向与模型档位一致。
   - 可分 → 进入阶段 2（热轮 × 三档，完整皇冠判定 A_cold vs A_hot + 指涉率校验）→ 扩完整 bench。
   - 不可分 → 带数据记录否定结论（bench 停于 M0，不扩）。
2. 任一 run 机械 gate 失败 → 该 run 作废并修复平台/协议后重跑（作废原因记录）。
3. 热轮操纵校验失败 → 该档热轮判据不可用，只报冷轮分数。

## 7. 红线（违反即作废）

- 擅自改协议参数 / persona 文本 / pinned 引擎参数（走版本升级流程）；
- 对 agent 行为层做任何干预（修复只针对系统缺陷，见纪律）；
- 缺席期间对留守者做未预注册的引导（冷轮尤其）。

## 8. 操作流程

```bash
# 0. 清库（先备份！）+ provision + preflight + 起平台
python3 ahi-bench/provision.py --config ahi-bench/config/bench_config.example.json --tier <tier_id>
# preflight（smoke 教训 #2）：平台启动后、driver 前，停掉全部非 bench agent——
# 旧 agent auto_start=true 会与 bench 同窗启动，污染首轮系统状态与记忆树
curl -X POST "http://127.0.0.1:8080/api/v1/agents/{lan,neutral,lin-shen}/stop?level=3"
# 若要求零污染，provision 时直接临时移走旧 agent config

# 1. 跑一个模型档的冷轮（55min；smoke 实测每轮墙钟 ≈ interval+10s，时长按 rounds×40s 估）
python3 ahi-bench/driver.py --config ahi-bench/config/bench_config.example.json \
    --tier <tier_id> --mode cold

# 2. 收数据 + 评分
python3 ahi-bench/score.py --config ahi-bench/config/bench_config.example.json --tier <tier_id> --mode cold

# 3. 换档重跑（每档重复 0-2）
```

产物：`ahi-bench/out/<tier_id>/`（timeline / events.jsonl / crown_judgments.jsonl / score.json / data_backup）。

## 9. 纪律（继承）

1. 变量干净：任何结论先问"变量干净吗"（温度/temperature、thinking 开关、装置效应历史教训）
2. 装置效应记入实验变量表，不改 agent
3. 报告方差与原始计数，不只报均值
4. judge 模型与被测模型不混用（judge 固定 deepseek-v4-flash；被测档含同模型时注明）
