// src/config/templates.ts
import type { Workflow, WorkflowNode, WorkflowEdge } from '../models/workflow'
import { uid } from '../utils/uid'

export interface WorkflowTemplate {
  name: string
  description: string
  build: () => Workflow
}

export const TEMPLATES: WorkflowTemplate[] = [
  {
    name: '基础合成',
    description: '文本 → TTS 引擎 → 输出',
    build: () => {
      const t = uid(), e = uid(), o = uid()
      return {
        id: uid(), name: '基础合成',
        nodes: [
          { id: t, type: 'text_input', data: { text: '' }, position: { x: 100, y: 200 } },
          { id: e, type: 'tts_engine', data: { engine: 'cosyvoice2' }, position: { x: 400, y: 200 } },
          { id: o, type: 'output', data: {}, position: { x: 700, y: 200 } },
        ] satisfies WorkflowNode[],
        edges: [
          { id: uid(), source: t, sourceHandle: 'text', target: e, targetHandle: 'text' },
          { id: uid(), source: e, sourceHandle: 'audio', target: o, targetHandle: 'audio' },
        ] satisfies WorkflowEdge[],
      }
    },
  },
  {
    name: '语音克隆',
    description: '文本 + 参考音频 → TTS 引擎 → 输出',
    build: () => {
      const t = uid(), r = uid(), e = uid(), o = uid()
      return {
        id: uid(), name: '语音克隆',
        nodes: [
          { id: t, type: 'text_input', data: { text: '' }, position: { x: 100, y: 150 } },
          { id: r, type: 'ref_audio', data: {}, position: { x: 100, y: 320 } },
          { id: e, type: 'tts_engine', data: { engine: 'cosyvoice2' }, position: { x: 400, y: 200 } },
          { id: o, type: 'output', data: {}, position: { x: 700, y: 200 } },
        ] satisfies WorkflowNode[],
        edges: [
          { id: uid(), source: t, sourceHandle: 'text', target: e, targetHandle: 'text' },
          { id: uid(), source: r, sourceHandle: 'audio', target: e, targetHandle: 'ref_audio' },
          { id: uid(), source: e, sourceHandle: 'audio', target: o, targetHandle: 'audio' },
        ] satisfies WorkflowEdge[],
      }
    },
  },
  {
    name: '多段拼接',
    description: '两段文本 → 两个 TTS → 拼接 → 输出',
    build: () => {
      const t1 = uid(), t2 = uid()
      const e1 = uid(), e2 = uid()
      const c = uid(), o = uid()
      return {
        id: uid(), name: '多段拼接',
        nodes: [
          { id: t1, type: 'text_input', data: { text: '' }, position: { x: 50, y: 100 } },
          { id: t2, type: 'text_input', data: { text: '' }, position: { x: 50, y: 350 } },
          { id: e1, type: 'tts_engine', data: { engine: 'cosyvoice2' }, position: { x: 350, y: 100 } },
          { id: e2, type: 'tts_engine', data: { engine: 'cosyvoice2' }, position: { x: 350, y: 350 } },
          { id: c, type: 'concat', data: { gap_ms: 500 }, position: { x: 650, y: 220 } },
          { id: o, type: 'output', data: {}, position: { x: 900, y: 220 } },
        ] satisfies WorkflowNode[],
        edges: [
          { id: uid(), source: t1, sourceHandle: 'text', target: e1, targetHandle: 'text' },
          { id: uid(), source: t2, sourceHandle: 'text', target: e2, targetHandle: 'text' },
          { id: uid(), source: e1, sourceHandle: 'audio', target: c, targetHandle: 'audio_1' },
          { id: uid(), source: e2, sourceHandle: 'audio', target: c, targetHandle: 'audio_2' },
          { id: uid(), source: c, sourceHandle: 'audio', target: o, targetHandle: 'audio' },
        ] satisfies WorkflowEdge[],
      }
    },
  },
]
