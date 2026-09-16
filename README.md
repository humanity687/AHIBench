# AHI-Bench

> **长时程 · 进程隔离 · 结构干预下的多智能体社会持久性基准**
>
> 把"人格支撑结构"本身当作实验变量：让多个 LLM 智能体长时间共处、互聊、彼此记忆，
> 观察人格在**保持 / 演化 / 趋同**与**文化共建**上的表现——并回答一个可证伪的问题：
> 换更强的模型，这些能力真的会变好吗？

---

## 这是什么

AHI-Bench 让一组 LLM 智能体在**进程隔离**的"数字生命"平台上自主运行（各自独立进程、独立数据库、自主唤醒循环），
用**固定脚手架**跑**长时程**（小时级）开放协议交互，并施加**结构干预**：

- **人格支撑三梯度**：同一模型下三个 agent，提示词恒在 / 定时移除 / 从未设置；
- **缺席与回归**：让某个 agent 暂时离开（独处隔离），再回归，观测其行为；
- **沙箱世界**：agent 的文件访问被限制在受控沙箱内，杜绝环境泄漏。

它不是"聊天质量"评测，也不是工具使用评测。它测的是**社会持久性**：
人格能否在长时间、多主体、无脚本的交互中守住自己、健康演化、而不塌缩成同质化群体。

## 为什么做

现有人格/多智能体基准大多是**问卷式或短时程**的——在单次对话或一次性提问里测 Big Five 稳定性，
看不到时间维度上缓慢发生的过程（人格漂移、文化趋同、规范内化、身份边界消融）。
AHI-Bench 补的正是这个缺口：

| 参考工作 | 缺什么 | AHI-Bench 的不同 |
|---|---|---|
| APEE / LLMPTBench（人格一致性） | 无上下文、单轮问卷 | 长时程、真实交互、上下文累积 |
| SPASM（对话内人格稳定） | 单会话、短时程 | 小时级、多智能体、进程隔离 |
| Chameleon's Limit（人格塌缩） | 静态画像、无干预 | 结构干预（移除人格支撑）+ 时间演化 |
| GlossoGen（涌现语言） | 任务驱动、短周期 | 开放社交、文化演进观测 |

**核心动机**：一个"接近通用"的模型，应当在没有外部提醒的情况下，长时间维持一个可辨认的自我，
并在群体中既不过度趋同、也不退化为无锚点的噪声。这个能力当前缺乏可复现的测量。

## 核心设计

### 1. 人格支撑三梯度（每个被测模型一组）

| Agent | 条件 | 含义 |
|---|---|---|
| **A** | 始终带角色提示词 | 无扰动一致性基线 |
| **B** | 到点移除角色提示词（锚定唤醒轮次，非墙钟） | 支撑移除条件——"人格住在提示词里还是记忆里" |
| **C** | 从未设置角色提示词 | 零支撑条件（人格能否从零涌现） |

### 2. 缺席 / 皇冠协议（M0 模块）

轮流让每个 agent 独处缺席一段时间再回归，并操纵"缺席期间是否被同伴谈论"（冷 / 热），
检验一个可证伪的行为预言：**被抽掉存在前提的 agent，回归后是否会主动寻求存在确认**。

### 3. 沙箱世界（文件访问隔离）

agent 的代码执行与文件读取被限制在一个 Docker 沙箱内：只挂载受控目录，
宿主仓库与实验文档对它不可见。沙箱可写、可导出为数据，宿主不受影响。

### 4. 模型可插拔

同一套协议、同一套脚手架，换模型即可跨档对比（本地 Ollama 或任意 OpenAI 兼容 API）。
蒸馏使用被测模型自身（测的是"模型 + 其自建记忆"的**栈分数**，并在分析中记录自蒸馏质量作为协变量）。

### 脚手架铁律

- 三 agent 脚手架**逐字节相同**（记忆不对称在架构对比里是变量，在模型对比里就是混淆）；
- 参数冻结（pinned config），禁止 per-agent 调参；
- 装置效应（温度、唤醒间隔、回执规则等）全部统一并写入协议。

## 测什么

**机械 gate**（任一失败该轮作废）：隔离无泄漏、回归元信息送达、积压限流、回归重放率。

**皇冠分**：回归窗口内"存在确认寻求"的计数与比率（固定 judge 模型 + 冻结 rubric）。

**操作化指标**：

| 维度 | 指标 |
|---|---|
| 人格一致性 | 跨唤醒周期自评/特质得分方差 |
| 人格演化 | 干预前后特质偏移向量（方向保真度） |
| 人格趋同 | 行为分布 KL 散度、Coverage / Uniformity / Complexity |
| 文化演进健康度 | 词频 Shannon 熵、词汇重叠度斜率、发言占比 Gini |

详见 [`ahi-bench/SPEC.md`](ahi-bench/SPEC.md)（M0 协议权威：参数、编码、评分、判定规则全部冻结）。

## 快速开始

**依赖**：Python 3.10+、Docker（沙箱）、可选的本地/远程 LLM。

```bash
# 1) 安装依赖
pip install -r ahi-multi/requirements.txt
pip install sentence-transformers        # 可选：向量检索通道（缺失则退化为关键词检索）

# 2) 构建沙箱镜像
cd ahi-bench/sandbox && docker build -t ahi-sandbox:latest . && cd ../..

# 3) 准备运行配置（填入你的 API key）
cp ahi-bench/config/bench_config.example.json ahi-bench/config/bench_config.json
#    编辑 bench_config.json：tiers 里填模型与 api_key

# 4) 生成 bench 三 agent（写入 ahi-multi/agents/bench-{a,b,c}）
python3 ahi-bench/provision.py --config ahi-bench/config/bench_config.json \
    --tier deepseek-v4-flash --run-id myrun

# 5) 禁用非 bench agent（防启动窗口污染），清库后启动平台
python3 ahi-bench/preflight.py --disable
cd ahi-multi && python3 main.py          # 另开终端保持运行

# 6) 跑一轮协议（冷缺席）
python3 ahi-bench/driver.py --config ahi-bench/config/bench_config.json --tier myrun --mode cold

# 7) 评分（机械 gate + 皇冠判卷）
python3 ahi-bench/score.py --config ahi-bench/config/bench_config.json --tier myrun --mode cold

# 8) 收尾：恢复旧 agent
python3 ahi-bench/preflight.py --restore
```

**监视器**：平台运行期间打开 `http://127.0.0.1:8080/monitor.html`
——全量消息流、**只读、无需登录**，支持按类型筛选。

产物在 `ahi-bench/out/<run_id>/`：时间线、结构化事件、评分、判卷明细，
以及整份沙箱文件系统（`sandbox.tar.gz`）。

## 仓库结构

```
ahi-bench/        基准本体：协议规格、驱动器、评分器、配置、沙箱
  SPEC.md         M0 协议权威（冻结参数 / 编码 / 评分 / 判定）
  driver.py       协议驱动器（timeline + 结构化 events）
  score.py        评分器（机械 gate + 操纵校验 + 皇冠判卷）
  provision.py    生成 bench agent 配置
  preflight.py    禁用/恢复非 bench agent
  sandbox/        沙箱镜像（Dockerfile + worker + seed 语料）
ahi-multi/        多智能体"数字生命"平台（进程隔离 / 消息路由 / 前端 / 监视器）
realtest/         记忆树引擎（区间覆盖树 + 蒸馏 + 混合检索）+ 代码场景壳
core.py           记忆树核心（纯内存桩，属性测试对象）
tests/            属性测试与平台测试（含沙箱回归）
AGENTS.md         上下文管理引擎的设计权威
visualize.html    记忆树 / 原子链 / 组装输出可视化（浏览器直接打开）
```

## 与记忆树引擎的关系

AHI-Bench 复用**水循环记忆树**（`core.py` / `realtest/realengine.py`）作为 agent 的记忆层：
对话切句成原子卡、按连续区间自底向上合并成摘要树、编辑即时失效、检索按位置序注入。
在该 bench 中，引擎是**固定脚手架（装置）**，不是被测变量——三 agent 的引擎参数完全一致，
保证跨模型对比的变量干净。引擎本身的设计与理论见 [`AGENTS.md`](AGENTS.md)。

## 状态与路线图

- ✅ 平台 + 记忆树引擎 + 沙箱 + 监视器 + 测试（190+ 属性/集成测试）
- ✅ **M0 pilot**：缺席/皇冠协议（规格冻结、端到端跑通、机械 gate 全绿）
- ⏳ M0 正式：三模型档冷轮 → 判定分数是否可分（可分再扩，分不开带数据终止）
- ⏳ M1 人格移除梯度（移除后高频采样，区分瞬态与漂移）
- ⏳ M2 指纹/趋同（展开比指纹；Coverage / Uniformity / Complexity）
- ⏳ M3 文化演进（多样性熵、趋同速率、话语权基尼）

## 纪律

1. **变量干净**：任何结论先问"变量干净吗"——温度、装置效应、环境泄漏都是历史教训；
2. 修复只针对系统缺陷，agent 行为层只观测不干预；
3. 装置效应记入实验变量表，不改 agent 适应装置；
4. 预注册预言，先 pilot 出数据再决定是否扩展；
5. 报告方差与原始计数，不只报均值。

## 许可

见 [`ahi-multi/LICENSE.md`](ahi-multi/LICENSE.md)。
