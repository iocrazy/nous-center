// 模型名校验(导入自愈,团队协作 incident:Infinite-Canvas 的 MiniMax_H3.json 冻结了
// 另一台机器上的模型文件名/子目录布局,导入本机后 ComfyUI 执行时才报 `Value not in
// list`,只能靠人肉改 JSON)。纯逻辑,不碰 React/网络 —— object_info 由调用方(
// ImportComfyDialog / ComfyTemplateEditor)已经拿到手,这里只做「哪些字段的值不在
// object_info 给的合法取值表里」的比对 + 「本机有没有像样的候选」的猜测,不发请求。
import type { ComfyWorkflow } from './comfyGraphLayout'

export interface ModelRefIssue {
  nodeId: string
  classType: string
  inputName: string
  /** 工作流里当前(不合法的)值。 */
  value: string
  /** object_info 给的该 class_type.inputName 合法取值全集。 */
  options: string[]
  /** 本机候选中猜出的最像的一个;猜不出则不给,交给用户手选。 */
  suggestion?: string
}

export interface ModelRefFix {
  nodeId: string
  inputName: string
  /** 要写入的**新**值。刻意不叫 `value` —— `ModelRefIssue.value` 是那个**坏**值,
   *  两者同名反义会让人把坏值原样传回来(实机验证时真踩过一次)。 */
  newValue: string
}

/** 一个 combo/enum 输入的声明形如 `[optionsArray, {…}?]` ——第一个元素是数组本身即取值表
 *  (跟 STRING/INT/FLOAT/BOOLEAN 那种 `["INT", {min,max,...}]` 用字符串占位类型名区分开)。
 *  跟 ComfyTemplateEditor.tsx 的 classInputDecl 同一套读法,这里独立一份是因为两边各自
 *  是纯函数/无状态组件,没有共享状态好抽,重复几行比强行共享一个内部 helper 更不容易
 *  在后续修改中彼此牵连。 */
function comboOptions(
  classType: string,
  inputName: string,
  objectInfo: Record<string, unknown>,
): string[] | null {
  const cls = objectInfo[classType] as
    | { input?: { required?: Record<string, unknown>; optional?: Record<string, unknown> } }
    | undefined
  const input = cls?.input
  const decl = input?.required?.[inputName] ?? input?.optional?.[inputName]
  if (!Array.isArray(decl)) return null
  const [typeOrOptions] = decl as [unknown, Record<string, unknown>?]
  if (!Array.isArray(typeOrOptions)) return null
  return (typeOrOptions as unknown[]).filter((o): o is string => typeof o === 'string')
}

function basename(path: string): string {
  const idx = path.lastIndexOf('/')
  return idx === -1 ? path : path.slice(idx + 1)
}

function stem(path: string): string {
  const base = basename(path)
  const idx = base.lastIndexOf('.')
  return idx === -1 ? base : base.slice(0, idx)
}

function tokenize(s: string): Set<string> {
  return new Set(s.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean))
}

/** 候选打分:先按 basename 精确匹配(子目录布局变了但文件名没变的最常见情况,例如
 *  `x.safetensors` vs `minimax-h3/x.safetensors`);basename 无匹配再退化成文件名 stem 的
 *  token 重合数最多者(`minimax_h3_ref2va_pruned_int8_convrot` 对 `minimax-h3/
 *  minimax_h3_ref2va_nvfp4`,共享 minimax/h3/ref2va 三个 token,压过完全不相关的选项)。
 *  两轮都定为「取分数最高、平分取 options 里先出现的」,确定性、可单测。 */
function bestSuggestion(value: string, options: string[]): string | undefined {
  const valueBase = basename(value)
  const exact = options.find((o) => basename(o) === valueBase)
  if (exact) return exact

  const valueTokens = tokenize(stem(value))
  if (valueTokens.size === 0) return undefined
  let best: string | undefined
  let bestScore = 0
  for (const o of options) {
    const oTokens = tokenize(stem(o))
    let score = 0
    for (const t of valueTokens) if (oTokens.has(t)) score++
    if (score > bestScore) {
      bestScore = score
      best = o
    }
  }
  return best
}

/** 扫描工作流每个节点的每个字符串输入,凡是 object_info 里声明为 combo/enum 且当前值
 *  不在合法取值表里的,记一条 issue。非 combo 的字符串输入(比如 prompt 文本)、
 *  object_info 里查不到的 class_type(自定义节点、或 sidecar 离线传 null/undefined)
 *  一律跳过,不误报。 */
export function findInvalidModelRefs(
  workflow: ComfyWorkflow,
  objectInfo: Record<string, unknown> | null | undefined,
): ModelRefIssue[] {
  if (!objectInfo) return []
  const issues: ModelRefIssue[] = []
  for (const [nodeId, node] of Object.entries(workflow)) {
    const classType = node?.class_type
    if (!classType || !objectInfo[classType]) continue
    for (const [inputName, rawValue] of Object.entries(node.inputs ?? {})) {
      if (typeof rawValue !== 'string') continue
      const options = comboOptions(classType, inputName, objectInfo)
      if (!options) continue
      if (options.includes(rawValue)) continue
      issues.push({
        nodeId,
        classType,
        inputName,
        value: rawValue,
        options,
        suggestion: bestSuggestion(rawValue, options),
      })
    }
  }
  return issues
}

/** issue 在 fixes/选择映射里的 key —— nodeId+inputName 唯一确定一个输入槽位。 */
export function issueKey(issue: Pick<ModelRefIssue, 'nodeId' | 'inputName'>): string {
  return `${issue.nodeId}::${issue.inputName}`
}

/** 按 fixes 把 workflow 里对应字段替换成用户选定的值,返回全新对象(深拷贝,不改入参)。
 *  workflow 来自解析后的 JSON,天然可结构化克隆,不用手写递归拷贝。 */
export function applyModelFixes(workflow: ComfyWorkflow, fixes: ModelRefFix[]): ComfyWorkflow {
  const next: ComfyWorkflow = JSON.parse(JSON.stringify(workflow))
  for (const fix of fixes) {
    const node = next[fix.nodeId]
    if (!node) continue
    node.inputs = { ...node.inputs, [fix.inputName]: fix.newValue }
  }
  return next
}
