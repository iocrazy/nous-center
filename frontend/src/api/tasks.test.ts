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
})
