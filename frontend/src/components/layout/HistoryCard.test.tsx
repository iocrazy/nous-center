import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import HistoryCard from './HistoryCard'
import { useExecutionStore } from '../../stores/execution'
import type { ExecutionTask } from '../../api/tasks'

// Arc 3(spec 2026-07-20-moss-asr §8):HistoryCard 的 asr 分支 —— 徽章「语音识别」+
// 摘要(时长/段数)+ failed 任务按 workflow_name 关键词兜底归 asr。
function mk(partial: Partial<ExecutionTask>): ExecutionTask {
  return {
    id: 'a1',
    workflow_id: null,
    workflow_name: 'moss-asr',
    status: 'completed',
    nodes_total: 0,
    nodes_done: 0,
    current_node: null,
    result: null,
    error: null,
    duration_ms: 1200,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    task_type: null,
    image_width: null,
    image_height: null,
    ...partial,
  } as ExecutionTask
}

describe('HistoryCard — ASR 分支(Arc 3)', () => {
  beforeEach(() => {
    useExecutionStore.setState({ expandedHistoryRowIds: new Set<string>() })
  })

  it('completed asr 任务显示「语音识别」徽章 + 时长/段数摘要', () => {
    const task = mk({
      type: 'asr',
      task_type: 'asr',
      audio_seconds: 7,
      segments_count: 2,
      result: { text: '你好世界', segments_count: 2, speakers: ['S01', 'S02'], audio_seconds: 7 },
    })
    render(<HistoryCard task={task} />)
    expect(screen.getByText('语音识别')).toBeTruthy()
    // collapsed 行摘要:7s · 2 段 · 1.2s(duration)
    expect(screen.getByText(/7s · 2 段/)).toBeTruthy()
  })

  it('failed asr 任务(无 result)按 workflow_name 关键词兜底归 asr', () => {
    const task = mk({
      workflow_name: 'moss-transcribe-diarize',
      status: 'failed',
      error: 'MOSS ASR 服务不可达',
      result: null,
    })
    render(<HistoryCard task={task} />)
    // failed 强制展开 → exp-head 的 TypeChip 显示「语音识别」
    expect(screen.getByText('语音识别')).toBeTruthy()
  })
})
