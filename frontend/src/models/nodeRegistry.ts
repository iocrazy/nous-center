import type { NodeType } from './workflow'

export type WidgetType = 'input' | 'textarea' | 'select' | 'slider' | 'agent_select'

export interface WidgetDef {
  name: string
  label: string
  widget: WidgetType
  options?: { value: string; label: string }[]
  min?: number
  max?: number
  step?: number
  precision?: number
  rows?: number
  default?: unknown
}

export interface DeclarativeNodeDef {
  type: NodeType
  label: string
  category: string
  badge: string
  badgeColor: string
  widgets: WidgetDef[]
}

export const DECLARATIVE_NODES: Record<string, DeclarativeNodeDef> = {
  llm: {
    type: 'llm',
    label: 'LLM',
    category: 'ai',
    badge: 'AI',
    badgeColor: 'var(--purple)',
    widgets: [
      { name: 'system', label: '系统提示', widget: 'textarea', rows: 3 },
      { name: 'model', label: '模型', widget: 'input' },
      { name: 'base_url', label: 'Base URL', widget: 'input', default: 'http://localhost:8100' },
      { name: 'api_key', label: 'API Key', widget: 'input' },
      { name: 'temperature', label: '温度', widget: 'slider', min: 0, max: 2, step: 0.1, precision: 1 },
      { name: 'max_tokens', label: '最大 Token', widget: 'slider', min: 1, max: 8192, step: 1, precision: 0 },
    ],
  },
  prompt_template: {
    type: 'prompt_template',
    label: '提示模板',
    category: 'ai',
    badge: 'AI',
    badgeColor: 'var(--purple)',
    widgets: [
      { name: 'template', label: '模板', widget: 'textarea', rows: 5 },
    ],
  },
  agent: {
    type: 'agent',
    label: 'Agent',
    category: 'ai',
    badge: 'AI',
    badgeColor: 'var(--purple)',
    widgets: [
      { name: 'agent_name', label: 'Agent', widget: 'agent_select' },
    ],
  },
  if_else: {
    type: 'if_else',
    label: '条件分支',
    category: 'control',
    badge: 'CTRL',
    badgeColor: 'var(--accent)',
    widgets: [
      { name: 'condition', label: '条件', widget: 'input' },
      {
        name: 'match_type',
        label: '匹配',
        widget: 'select',
        options: [
          { value: 'contains', label: '包含' },
          { value: 'equals', label: '等于' },
          { value: 'regex', label: '正则' },
          { value: 'not_empty', label: '非空' },
        ],
      },
    ],
  },
}

export interface NodeCategoryDef {
  name: string
  label: string
  color: string
  nodes: { type: NodeType; dotColor: string }[]
}

export const NODE_CATEGORIES: NodeCategoryDef[] = [
  {
    name: 'ai',
    label: 'AI',
    color: 'var(--purple)',
    nodes: [
      { type: 'llm', dotColor: 'var(--purple)' },
      { type: 'prompt_template', dotColor: 'var(--purple)' },
      { type: 'agent', dotColor: 'var(--purple)' },
    ],
  },
  {
    name: 'control',
    label: '控制流',
    color: 'var(--accent)',
    nodes: [
      { type: 'if_else', dotColor: 'var(--accent)' },
    ],
  },
]
