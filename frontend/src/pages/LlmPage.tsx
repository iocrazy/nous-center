/**
 * LLM page placeholder(PR-2,任务面板重置)。占 `/llm` 路由,真 chat UI 在后续 PR。
 */
import { MessageSquare } from 'lucide-react'

export default function LlmPage() {
  return (
    <div
      className="flex-1 flex flex-col items-center justify-center"
      style={{ background: 'var(--tp-bg-base)', color: 'var(--tp-text)' }}
    >
      <MessageSquare size={32} style={{ color: 'var(--type-llm)' }} />
      <h1 className="mt-4 text-2xl font-semibold">LLM</h1>
      <p className="mt-1 text-sm" style={{ color: 'var(--tp-text-muted)' }}>对话 · 即将上线</p>
    </div>
  )
}
