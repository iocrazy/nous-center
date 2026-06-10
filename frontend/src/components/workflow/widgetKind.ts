// 节点属性弹窗的「控件类型」下拉(复刻 Infinite-Canvas 的类型选择)。
// nous 的 type/constraints 本由强类型节点注册表派生,这里允许在发布时**覆盖**
// 成另一种控件(如把数字改成纯输入、把字符串改成下拉)。每个 kind ⇄ 一组
// {type, constraints},与 SchemaDrivenForm 的 controlKind 判定保持一致。
import type { ExposedParam } from '../../api/services'

export type WidgetKind =
  | 'text'
  | 'textarea'
  | 'number'
  | 'integer'
  | 'slider'
  | 'boolean'
  | 'select'
  | 'image'

export const KIND_OPTIONS: Array<{ v: WidgetKind; label: string }> = [
  { v: 'text', label: '文本' },
  { v: 'textarea', label: '多行文本' },
  { v: 'number', label: '数字' },
  { v: 'integer', label: '整数' },
  { v: 'slider', label: '滑杆' },
  { v: 'boolean', label: '开关' },
  { v: 'select', label: '下拉' },
  { v: 'image', label: '图片' },
]

function c(p: ExposedParam): Record<string, unknown> {
  return (p.constraints ?? {}) as Record<string, unknown>
}

/** 当前 param 对应哪个 kind —— 与 SchemaDrivenForm.controlKind 同口径。 */
export function kindOf(p: ExposedParam): WidgetKind {
  const cons = c(p)
  const t = String(p.type ?? 'string').toLowerCase()
  if (Array.isArray(cons.enum) && cons.enum.length > 0) return 'select'
  const numeric = t === 'integer' || t === 'int' || t === 'number' || t === 'float'
  if (numeric && typeof cons.min === 'number' && typeof cons.max === 'number') return 'slider'
  if (t === 'integer' || t === 'int') return 'integer'
  if (t === 'number' || t === 'float') return 'number'
  if (t === 'boolean' || t === 'bool') return 'boolean'
  if (t === 'file' || t === 'image' || t === 'audio' || t === 'video' || t === 'binary') return 'image'
  return cons.format === 'single_line' ? 'text' : 'textarea'
}

/** 把 param 切到指定 kind,尽量保留有用的原约束(下拉的 enum / 滑杆的 min/max/step)。 */
export function applyKind(p: ExposedParam, kind: WidgetKind): ExposedParam {
  const cons = c(p)
  const keepEnum = Array.isArray(cons.enum) ? { enum: cons.enum, ...(cons.enum_labels ? { enum_labels: cons.enum_labels } : {}) } : {}
  const range: Record<string, unknown> = {}
  if (typeof cons.min === 'number') range.min = cons.min
  if (typeof cons.max === 'number') range.max = cons.max
  if (typeof cons.step === 'number') range.step = cons.step
  switch (kind) {
    case 'text':
      return { ...p, type: 'string', constraints: { format: 'single_line' } }
    case 'textarea':
      return { ...p, type: 'string', constraints: {} }
    case 'number':
      return { ...p, type: 'number', constraints: {} }
    case 'integer':
      return { ...p, type: 'integer', constraints: {} }
    case 'slider': {
      const type = p.type === 'integer' || p.type === 'int' ? 'integer' : 'number'
      const r = Object.keys(range).length ? range : { min: 0, max: 100, step: 1 }
      return { ...p, type, constraints: r }
    }
    case 'boolean':
      return { ...p, type: 'boolean', constraints: {} }
    case 'select':
      return { ...p, type: 'string', constraints: Object.keys(keepEnum).length ? keepEnum : { enum: [] } }
    case 'image':
      return { ...p, type: 'image', constraints: {} }
  }
}
