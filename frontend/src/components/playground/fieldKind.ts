import type { ExposedParam } from '../../api/services'
import { paramKey, paramSlot } from '../../api/services'

/** 字段类型判定与默认值推断 —— 从 SchemaDrivenForm.tsx 抽出来的纯函数。
 *
 * 为什么单独成文件:`ComfyTemplateEditor` 要复用 `defaultFor`(算「画布节点预览」
 * 左栏的字段初值),但从组件文件里导出非组件会打破 React Fast Refresh
 * (eslint `react-refresh/only-export-components`,CI 上是 error)。
 * 纯逻辑挪到这里,组件文件只导出组件。
 */
export type FieldKind =
  | 'string'
  | 'string_multiline'
  | 'number'
  | 'integer'
  | 'boolean'
  | 'file'
  | 'select'
  | 'thumb_select'
  | 'slider'

export function num(v: unknown): number | undefined {
  const n = Number(v)
  return Number.isFinite(n) ? n : undefined
}

export function classifyField(p: ExposedParam): FieldKind {
  const constraints = (p.constraints ?? {}) as Record<string, unknown>
  const t = (p.type ?? 'string').toLowerCase()
  // 文件类语义优先于 enum:ComfyUI 的 LoadImage 等节点经 /object_info 带出的
  // enum 是 sidecar 上已存在的文件名列表(如 ["5 (1).jpg","example.png"]),不是
  // 一个可选值域 —— 桥会把上传的 data URI 传给 sidecar 落盘,不是从这批名字里选一个。
  // 必须在 enum 判断之前拦掉,否则这类字段会被误判成下拉框,Playground 就没有上传控件。
  if (t === 'file' || t === 'image' || t === 'audio' || t === 'video' || t === 'binary' || t === 'media') {
    return 'file'
  }
  if (Array.isArray(constraints.enum) && constraints.enum.length > 0) {
    // option_meta 里只要有任何一项带 image,就用缩略图网格 —— 275 个 fooocus 风格
    // 用纯文本下拉根本没法挑(后端 comfy_templates.py::_split_options 写入)。
    // 一张图都没有时照旧退回 <select>,不做无谓的布局变更。
    const meta = constraints.option_meta
    if (Array.isArray(meta) && meta.some((m) => (m as { image?: unknown })?.image)) {
      return 'thumb_select'
    }
    return 'select'
  }
  const numeric = t === 'integer' || t === 'int' || t === 'number' || t === 'float'
  // Numeric field with a bounded range → slider (对齐 Infinite-Canvas / nous
  // 节点的 slider widget,min/max/step 从 ExposedParam.constraints 带出)。
  if (numeric && num(constraints.min) !== undefined && num(constraints.max) !== undefined) {
    return 'slider'
  }
  if (t === 'integer' || t === 'int') return 'integer'
  if (t === 'number' || t === 'float') return 'number'
  if (t === 'boolean' || t === 'bool') return 'boolean'
  // Strings default to multiline. Single-line is opt-in via constraints.format='single_line'
  // — published service inputs are almost always free text (prompts, transcripts,
  // user content) and a single 32-char input is unusable for those.
  if (constraints.format === 'single_line') return 'string'
  return 'string_multiline'
}

/** Seed-like numeric field → show a 🎲 randomize button. Opt-in via
 *  constraints.random, or auto-detected by a `seed` key/slot name. */
export function isRandomizable(p: ExposedParam): boolean {
  const constraints = (p.constraints ?? {}) as Record<string, unknown>
  if (constraints.random === true) return true
  const name = `${paramKey(p) ?? ''} ${paramSlot(p) ?? ''}`.toLowerCase()
  return /seed/.test(name)
}

/** exported: ComfyTemplateEditor 的「画布节点预览」左栏复用它算字段初值,不用重复
 *  一套 default 推断逻辑(该逻辑 defaultFor 就已经跟 classifyField 的字段类型判定绑死)。 */
export function defaultFor(p: ExposedParam): unknown {
  if (p.default !== undefined && p.default !== null) return p.default
  switch (classifyField(p)) {
    case 'boolean':
      return false
    case 'slider': {
      const c = (p.constraints ?? {}) as Record<string, unknown>
      return num(c.min) ?? 0
    }
    case 'number':
    case 'integer':
      return ''
    case 'file':
      return null
    default:
      return ''
  }
}

