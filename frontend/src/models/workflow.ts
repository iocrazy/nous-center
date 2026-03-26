export type PortType = 'text' | 'audio' | 'message' | 'data' | 'any'

/** Node type identifier. Built-in types are listed below; plugin packages can add more at runtime. */
export type NodeType = string

/** Built-in node types (for reference / type narrowing). */
export type BuiltinNodeType =
  | 'text_input'
  | 'ref_audio'
  | 'tts_engine'
  | 'resample'
  | 'mixer'
  | 'concat'
  | 'bgm_mix'
  | 'output'
  | 'llm'
  | 'prompt_template'
  | 'agent'
  | 'if_else'
  | 'python_exec'

export interface PortDef {
  id: string
  type: PortType
  label: string
}

export interface NodeDef {
  type: NodeType
  label: string
  inputs: PortDef[]
  outputs: PortDef[]
}

export const NODE_DEFS: Record<NodeType, NodeDef> = {
  text_input: {
    type: 'text_input',
    label: '文本输入',
    inputs: [],
    outputs: [{ id: 'text', type: 'text', label: '文本' }],
  },
  ref_audio: {
    type: 'ref_audio',
    label: '参考音频',
    inputs: [],
    outputs: [{ id: 'audio', type: 'audio', label: '音频' }],
  },
  tts_engine: {
    type: 'tts_engine',
    label: 'TTS 引擎',
    inputs: [
      { id: 'text', type: 'text', label: '文本' },
      { id: 'ref_audio', type: 'audio', label: '参考音频' },
    ],
    outputs: [{ id: 'audio', type: 'audio', label: '音频' }],
  },
  resample: {
    type: 'resample',
    label: '重采样',
    inputs: [{ id: 'audio', type: 'audio', label: '输入' }],
    outputs: [{ id: 'audio', type: 'audio', label: '输出' }],
  },
  mixer: {
    type: 'mixer',
    label: '混音',
    inputs: [
      { id: 'audio_1', type: 'audio', label: '轨道 1' },
      { id: 'audio_2', type: 'audio', label: '轨道 2' },
    ],
    outputs: [{ id: 'audio', type: 'audio', label: '输出' }],
  },
  concat: {
    type: 'concat',
    label: '拼接',
    inputs: [
      { id: 'audio_1', type: 'audio', label: '音频 1' },
      { id: 'audio_2', type: 'audio', label: '音频 2' },
    ],
    outputs: [{ id: 'audio', type: 'audio', label: '输出' }],
  },
  bgm_mix: {
    type: 'bgm_mix',
    label: 'BGM 混合',
    inputs: [
      { id: 'speech', type: 'audio', label: '语音' },
      { id: 'bgm', type: 'audio', label: 'BGM' },
    ],
    outputs: [{ id: 'audio', type: 'audio', label: '输出' }],
  },
  output: {
    type: 'output',
    label: '输出播放',
    inputs: [{ id: 'audio', type: 'audio', label: '音频' }],
    outputs: [],
  },
  llm: {
    type: 'llm',
    label: 'LLM',
    inputs: [{ id: 'prompt', type: 'text', label: '提示' }],
    outputs: [{ id: 'text', type: 'text', label: '输出' }],
  },
  prompt_template: {
    type: 'prompt_template',
    label: '提示模板',
    inputs: [{ id: 'text', type: 'text', label: '输入' }],
    outputs: [{ id: 'text', type: 'text', label: '输出' }],
  },
  agent: {
    type: 'agent',
    label: 'Agent',
    inputs: [{ id: 'text', type: 'text', label: '输入' }],
    outputs: [{ id: 'text', type: 'text', label: '文本' }],
  },
  if_else: {
    type: 'if_else',
    label: '条件分支',
    inputs: [{ id: 'input', type: 'text', label: '输入' }],
    outputs: [
      { id: 'true', type: 'text', label: '真' },
      { id: 'false', type: 'text', label: '假' },
    ],
  },
  python_exec: {
    type: 'python_exec',
    label: 'Python 执行',
    inputs: [{ id: 'text', type: 'text', label: '输入' }],
    outputs: [{ id: 'text', type: 'text', label: '输出' }],
  },
}

export interface WorkflowNode {
  id: string
  type: NodeType
  data: Record<string, unknown>
  position: { x: number; y: number }
}

export interface WorkflowEdge {
  id: string
  source: string
  sourceHandle: string
  target: string
  targetHandle: string
}

export interface Workflow {
  id: string
  name: string
  description?: string
  nodes: WorkflowNode[]
  edges: WorkflowEdge[]
  is_template?: boolean
  status?: 'draft' | 'published'
}
