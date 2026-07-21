import { describe, it, expect } from 'vitest'
import type { ExecutionTask } from './tasks'

describe('ExecutionTask V1.5 fields', () => {
  it('accepts V1.5 scheduler + thumbnail fields', () => {
    // 编译期断言：下面这个对象能赋给 ExecutionTask 就说明接口已扩。
    const t: ExecutionTask = {
      id: 'wf_1',
      workflow_id: null,
      workflow_name: 'flux2-人物立绘',
      status: 'completed',
      nodes_total: 2,
      nodes_done: 2,
      current_node: null,
      result: null,
      error: null,
      duration_ms: 34000,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      task_type: 'image',
      image_width: 1024,
      image_height: 1024,
      gpu_group: 'image',
      runner_id: 'runner-i',
      queue_position: null,
      output_thumbnails: ['/files/outputs/wf_1/0.webp'],
    }
    expect(t.output_thumbnails?.[0]).toContain('outputs')
    expect(t.gpu_group).toBe('image')
  })

  it('V1.5 fields are optional (old backend payload still valid)', () => {
    const legacy: ExecutionTask = {
      id: 'wf_2',
      workflow_id: null,
      workflow_name: 'legacy',
      status: 'queued',
      nodes_total: 0,
      nodes_done: 0,
      current_node: null,
      result: null,
      error: null,
      duration_ms: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      task_type: null,
      image_width: null,
      image_height: null,
    }
    expect(legacy.gpu_group).toBeUndefined()
  })

  it('Arc 3: accepts asr task_type + audio_seconds / segments_count', () => {
    const asr: ExecutionTask = {
      id: 'act_1',
      workflow_id: null,
      workflow_name: 'moss-asr',
      status: 'completed',
      nodes_total: 0,
      nodes_done: 0,
      current_node: null,
      result: { text: '你好世界', segments_count: 2, speakers: ['S01', 'S02'], audio_seconds: 7 },
      error: null,
      duration_ms: 800,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      task_type: 'asr',
      type: 'asr',
      image_width: null,
      image_height: null,
      audio_seconds: 7,
      segments_count: 2,
    }
    expect(asr.task_type).toBe('asr')
    expect(asr.audio_seconds).toBe(7)
    expect(asr.segments_count).toBe(2)
  })
})
