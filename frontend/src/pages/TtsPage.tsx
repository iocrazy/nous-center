/**
 * TTS page placeholder(PR-2,任务面板重置)。占 `/tts` 路由,真 UI 在后续 PR。
 */
import { Mic } from 'lucide-react'

export default function TtsPage() {
  return (
    <div
      className="flex-1 flex flex-col items-center justify-center"
      style={{ background: 'var(--tp-bg-base)', color: 'var(--tp-text)' }}
    >
      <Mic size={32} style={{ color: 'var(--type-tts)' }} />
      <h1 className="mt-4 text-2xl font-semibold">TTS</h1>
      <p className="mt-1 text-sm" style={{ color: 'var(--tp-text-muted)' }}>语音合成 · 即将上线</p>
    </div>
  )
}
