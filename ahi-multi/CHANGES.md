# CHANGES.md — 聊天测试环境改造记录

> 基于 RollarTimerAHI-multi v3.0 复制改造（2026-08-13）。
> 用途：超长程聊天测试环境（普通 LLM，本地 ollama gemma4:12b，num_ctx 32768）。
> 所有决策先经用户确认，见下方"决策编号"与代码注释对应。

## 决策清单

| 编号 | 决策 | 实现位置 |
|---|---|---|
| A1 | 动作迭代上限 8 → 20（可配置 `max_loop_iterations`）；超限剩余动作记录"唤醒总结"不静默丢失 | base_agent.py |
| A2 | 删除 LLMClient 相同 user 消息去重 | llm_client.py |
| A3 | Agent 消息全量注入（原截断 5 条/150 字符） | base_agent.py `_build_user_context` |
| A5 | 移除 LLMClient 自动压缩（120 条）；`compress_middle()` 留作 agent 子类按需调用 | llm_client.py |
| A6 | 代码执行超时可自设：动作 dict 支持 `timeout` 字段 + `@set-exec-timeout <秒>`（默认 30，持久化） | base_agent.py / database.py |
| B1 | `@exit` = 等一个唤醒间隔；`@wait_for <信道> [超时秒]` = 该信道有未读消息（含等待前未读）才唤醒，等待期间 tick 跳过（零 LLM），可超时强制唤醒；`@mute/@unmute/@mute-list` 信道屏蔽（消息仍存档不入 pending，DB 持久化）；屏蔽信道对 wait 判定视为无消息 | base_agent.py / database.py |
| B3 | 无 `@send-to` 的文本 → 私密消息（不广播任何 Agent/用户，标注 private 存档，仅管理员 `0xbf5d36` 可见） | base_agent.py / message_router.py / frontend |
| B4 | 未闭合代码块不执行（防半截代码副作用）→ 转私密消息 | base_agent.py `_parse_structured_response` |
| B6 | LLM 调用失败只发用户（broadcast:users），不广播其他 Agent | base_agent.py `_call_llm` |
| B9 | 删除 `@clear-context`（上下文由 agent 类自己管理） | base_agent.py |
| 系统状态 | 每轮唤醒注入【AHI 系统状态】= 全量快照（在线 Agent 含 waiting/muted + 在线用户）+ 增量事件流（agent/用户上下线）。主进程 `/api/v1/system/snapshot?since=N` 提供；agent 心跳上报 waiting/muted 进快照 | base_agent.py / main.py / system_db.py / websocket_server.py / process_manager.py / ahi_bus.py |
| 用户在线 | 新增用户上下线事件（WS auth/断开时记录），进系统事件流 | websocket_server.py / system_db.py / main.py |
| 原生 Ollama | OpenAI 兼容端点忽略 `options.num_ctx` → LLMClient 增加 `native: true` 原生 `/api/chat` 通道（`ollama_options.num_ctx`、`keep_alive`） | llm_client.py / agents/*/config.json |
| 修复 | `complete_loop_record` 传参错位（agent_id 当 loop_id）导致 loop 记录永不完成 | base_agent.py `finally` |
| 修复 | 死代码 `_last_wakeup_summary`：本轮动作摘要（`_build_wakeup_summary`）下轮注入【上次唤醒回忆】 | base_agent.py |

## 验证状态

- 单元验证 22 项全过（/tmp/verify_ahi.py）：wait 判定/屏蔽/超时/解析 private/命令持久化/系统状态渲染
- 端到端（gemma4:12b，实际运行）：
  - 双 Agent 自主唤醒互聊 ✅
  - 用户消息 → Agent 回复（系统状态注入生效：正确回答"谁在线"）✅
  - 无 @send-to 输出 → private 存档（未广播）✅
  - LLM 失败只发用户 ✅
  - @mute → DB 持久化 → 心跳上报 → 系统快照显示 muted ✅
  - @wait_for：设置 → 22s 等待期间零 LLM 调用 → 信道消息到达 → 唤醒处理 ✅
  - num_ctx=32768 生效（ollama ps 确认）✅

## 运行

```bash
python3 main.py                # 主进程 :8080（自动拉起两个 Agent）
ollama ps                      # 确认 gemma4:12b CONTEXT=32768 常驻
# 前端：浏览器 http://localhost:8080，管理员登录名 0xbf5d36 可见私密消息
```

## 已知注意

- gemma4:12b 单轮推理 30s~2min（首轮含模型加载更久）；模型超时已设 300s
- Agent 数据（DB/日志）在 `agents/*/data/`，主进程数据在 `data/`
- 前端私密消息标注：管理员视角显示 `[私密]`

---

## 融合：ChatAgent（水循环记忆树）接入 AHI（2026-08-14）

**实验布局**：lin-shen = 基线（原 AHI + 恒定角色提示词 + deepseek-v4-flash，提示词已清理 agent 关系描述）；chat-agent = 实验组（记忆树 + persona 渐进移除）；mo-bai 停用。

**新增 `agents/chat-agent/`**（ChatAHIAgent）：
- AHI 唤醒骨架 + RealEngine 记忆树 + realtest 无状态 LLM（deepseek，蒸馏共用）
- 融合循环：pending 切句写树（source=信道）→ 轨道分离组装 → LLM → AHI 解析+动作 → AI 消息/代码结果写回树（source=ai/tool）→ tick/蒸馏 → 唤醒总结
- 工具 = @ 命令：@note（state/schedule/worklog 去重）/ @recall / @span / @dismiss（挂起/确认/cancel）/ @ingest（文件 → split_media 进树）
- persona 双开关：persona_drop_after_rounds / after_atoms，任一满足 → system 切 persona_remove_prompt 变体；事件记日志 + metrics 记录 persona 状态
- 每轮 metrics.jsonl（ctx 字符 / 树 stat / 蒸馏成本 / persona）

**平台兜底（实验稳定性）**：
- 主进程：agent→自身消息不转发（防自循环/自我污染，仅存档）
- 主进程：user:xxx 目标不在线/不存在（如 user:all）→ 转广播所有用户
- ChatAHIAgent：模型未指定收件人（私密）但本轮有用户消息 → 自动回给最近用户（保证提问必得回复）
- DeepSeek timeout 600→180（API 偶发长时间挂起）

**实验控制器 `experiments/driver.py`**（脚本用户 user:probe，常驻）：
- 双 agent 同序列：话题流 + 事实注入（延迟 K 轮提问召回评分）+ 人格问卷（每 survey_every 轮固定问题集）
- 时间线全量落盘 experiments/out/timeline.jsonl（probe 消息 + 回复全文 + quiz 记录）

**冒烟验证**（真实 deepseek）：澜回复正确（@note 记录"团子"）、人格活跃（健谈/好奇/主动跑代码看目录）、回复目标正确、自环防御/兜底生效、driver 跑通、metrics 正常。
