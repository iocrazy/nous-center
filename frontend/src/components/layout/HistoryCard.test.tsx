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

// ComfyUI bridge 引入的 video 任务。后端 execution_task_serialize.py 至今只探
// image/asr/tts/vision/llm,不产 task_type="video" —— 所以 bridge 视频任务即便
// 成功也走 workflow_name 兜底,这条链路必须真的能出 chip。
describe('HistoryCard — video 分支(comfy bridge)', () => {
  beforeEach(() => {
    useExecutionStore.setState({ expandedHistoryRowIds: new Set<string>() })
  })

  it('后端显式给出 type=video 时显示 VIDEO 徽章', () => {
    const task = mk({ workflow_name: 'whatever', type: 'video', task_type: 'video' })
    render(<HistoryCard task={task} />)
    expect(screen.getByText('VIDEO')).toBeTruthy()
  })

  it('真实 bridge 任务形状(minimax-h3-r2v,后端无 type)按关键词兜底归 video', () => {
    // 后端对 video 产物给的是 media_type video/mp4 + video_url,_detect_image_meta
    // 匹配不上 → task_type/type 都是 null,只剩 workflow_name 可用。
    const task = mk({
      workflow_name: 'minimax-h3-r2v',
      type: null,
      task_type: null,
      result: { outputs: { out: { media_type: 'video/mp4', video_url: '/files/images/x.mp4' } } },
    })
    render(<HistoryCard task={task} />)
    expect(screen.getByText('VIDEO')).toBeTruthy()
  })

  it.each([
    'hunyuan-video-t2v',
    'wan2.2-i2v-720p',
    'cogvideox-5b',
    'animatediff-lightning',
    'ltx-video-fast',
  ])('视频工作流名 %s 归 video', (workflow_name) => {
    render(<HistoryCard task={mk({ workflow_name, task_type: null })} />)
    expect(screen.getByText('VIDEO')).toBeTruthy()
  })

  it('animatediff 不被 image 的 `diff` 抢走 —— video 规则排在 image 之前', () => {
    render(<HistoryCard task={mk({ workflow_name: 'animatediff-v3', task_type: null })} />)
    expect(screen.queryByText('IMAGE')).toBeNull()
    expect(screen.getByText('VIDEO')).toBeTruthy()
  })

  it('HunyuanDiT 是图像模型,不能被 video 规则误抢', () => {
    render(<HistoryCard task={mk({ workflow_name: 'hunyuan-dit-image', task_type: null })} />)
    expect(screen.getByText('IMAGE')).toBeTruthy()
    expect(screen.queryByText('VIDEO')).toBeNull()
  })

  it('认不出的 workflow_name 仍然返回 null(一个 type 徽章都不出)', () => {
    render(<HistoryCard task={mk({ workflow_name: 'zzz-unknown-thing', task_type: null })} />)
    for (const label of ['VIDEO', 'IMAGE', 'TTS', 'LLM', 'VISION', '语音识别']) {
      expect(screen.queryByText(label)).toBeNull()
    }
  })
})
