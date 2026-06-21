import { describe, expect, it } from 'vitest'

import { endpointFor } from './services'

describe('endpointFor — 服务卡端点提示按 category', () => {
  const ep = (category: string | null) => endpointFor({ name: 's', category: category as never })

  it('每个 model 类目落对的 OpenAI 兼容端点(2026-06-21:此前非 llm 全误落 /v1/apps)', () => {
    expect(ep('llm')).toContain('/v1/chat/completions')
    expect(ep('embedding')).toContain('/v1/embeddings')
    expect(ep('tts')).toContain('/v1/audio/speech')
    expect(ep('asr')).toContain('/v1/audio/transcriptions')
    expect(ep('image')).toContain('/v1/images/generations')
  })

  it('app/workflow(无 model 类目)走 /v1/apps/.../run', () => {
    expect(ep('app')).toContain('/v1/apps/s/run')
    expect(ep(null)).toContain('/v1/apps/s/run')
  })

  it('asr 不串到 tts 的 speech 端点', () => {
    expect(ep('asr')).not.toContain('/v1/audio/speech')
  })
})
