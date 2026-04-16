# LLM 节点 Token Stats 实时显示

## Context

工作流编辑器的 LLM 节点（DeclarativeNode）执行时没有 token 用量和速率反馈。
用户需要在节点卡片上看到：生成中的实时速率 + 完成后的最终汇总。

## 决策

1. **展示位置**：LLM 节点卡片底部，现有 widget 行下方，只在执行中或有结果时显示。
2. **生成中**：`⚡ 生成中 · 78 tok/s · 输出 234`（250ms throttle 刷新）。
3. **完成后**：`✓ 输入 245 · 输出 1024 · 合计 1269 · 78 tok/s · 12.9s`（定格，直到下次执行或 3s 后淡出）。
4. **速率算法**：排除首 token（TTFT 不是生成速率），`rate = (tokenCount - 1) / elapsed_since_first_token`。
5. **数据来源**：后端 `node_complete` 事件增加 `usage` + `duration_ms` 字段。

## 前端改动

### DeclarativeNode.tsx

新增 state：
```typescript
const [tokenStats, setTokenStats] = useState<{
  phase: 'streaming' | 'done'
  outputTokens: number
  inputTokens: number
  totalTokens: number
  tokensPerSec: number
  durationSec: number
} | null>(null)
```

流式期间：
- 复用已有的 `node-progress` window event listener
- `node_stream` 事件：`tokenCount++`，首次记录 `firstTokenAt = performance.now()`
- 每 250ms throttle 算一次 rate 并 setState
- `node_complete` 带 `usage` → 用最终值覆盖，phase 切 'done'

渲染：
- 卡片底部新增一行 `<TokenStatsBar />`
- 字号 9，颜色 `var(--muted)`
- 图标用 lucide `Zap`（生成中）/ `Check`（完成）
- phase='done' 时 3s 后淡出（opacity transition）

### 节点宽度

不增加固定高度。stats 行仅在 `tokenStats !== null` 时渲染，其余时间不占空间。

## 后端改动

### workflow_executor.py

`_execute_node` 返回的 output dict 如果包含 `usage` key，将其透传到 `node_complete` 事件：

```python
# 现有代码 (line ~173)
if self._on_progress:
    await self._on_progress({
        "type": "node_complete",
        "node_id": node_id,
        "step": i + 1,
        "total": total,
        "progress": round(((i + 1) / total) * 100),
        # 新增：
        "usage": output.get("usage"),
        "duration_ms": output.get("duration_ms"),
    })
```

### LLM 节点 executor

LLM 节点执行器（前端 `workflowExecutor.ts` 里的 `llm` executor）调用 `/v1/chat/completions` 后，需要把 `response.usage` 写入 output：

```typescript
// nodeExecutors['llm'] 里
const usage = response.usage  // {prompt_tokens, completion_tokens, total_tokens}
return { text: content, usage, duration_ms: elapsed }
```

后端 plugin 节点同理：executor 返回 output 时附带 usage。

## 速率精度说明

- 前端 token 计数基于 `node_stream` 事件数（每个 SSE chunk 算 1 token）
- 实际 vLLM 可能一个 chunk 含多 token（rare but possible）
- 最终 `node_complete.usage.completion_tokens` 是真值，速率 = `completion_tokens / duration_sec`
- 流式期间的速率是近似值，完成后用真值覆盖

## 不做的事

- 不做历史统计（已有 Dashboard + /usage/inference API）
- 不做非 LLM 节点的 stats（TTS 节点可未来扩展）
- 不做 TTFT（首 token 延迟）单独展示（可未来加 tooltip）
