import type { Workflow, WorkflowNode, WorkflowEdge, NodeType } from '../models/workflow'
import { apiFetch } from '../api/client'
import type { SynthesizeResponse } from '../api/tts'

export interface ExecutionResult {
  audioBase64: string
  sampleRate: number
  duration: number
}

// Each node produces typed output during execution
type NodeOutput = {
  text?: string
  audioBase64?: string
  sampleRate?: number
  audioPath?: string
}

function topoSort(nodes: WorkflowNode[], edges: WorkflowEdge[]): WorkflowNode[] {
  const inDegree = new Map<string, number>()
  const adj = new Map<string, string[]>()

  for (const n of nodes) {
    inDegree.set(n.id, 0)
    adj.set(n.id, [])
  }

  for (const e of edges) {
    inDegree.set(e.target, (inDegree.get(e.target) ?? 0) + 1)
    adj.get(e.source)?.push(e.target)
  }

  const queue: string[] = []
  for (const [id, deg] of inDegree) {
    if (deg === 0) queue.push(id)
  }

  const sorted: WorkflowNode[] = []
  const nodeMap = new Map(nodes.map((n) => [n.id, n]))

  while (queue.length > 0) {
    const id = queue.shift()!
    const node = nodeMap.get(id)
    if (node) sorted.push(node)

    for (const next of adj.get(id) ?? []) {
      const deg = (inDegree.get(next) ?? 1) - 1
      inDegree.set(next, deg)
      if (deg === 0) queue.push(next)
    }
  }

  if (sorted.length !== nodes.length) {
    throw new Error('工作流存在循环依赖')
  }

  return sorted
}

function getInputs(
  nodeId: string,
  edges: WorkflowEdge[],
  outputs: Map<string, NodeOutput>,
): NodeOutput {
  const merged: NodeOutput = {}
  for (const e of edges) {
    if (e.target === nodeId) {
      const src = outputs.get(e.source)
      if (!src) continue
      // Map source output to target input by handle type
      if (e.targetHandle === 'text' && src.text) merged.text = src.text
      if (e.targetHandle?.startsWith('audio') && src.audioBase64) {
        merged.audioBase64 = src.audioBase64
        merged.sampleRate = src.sampleRate
      }
      if (e.targetHandle === 'ref_audio' && src.audioBase64) {
        merged.audioBase64 = src.audioBase64
      }
      // For multi-input nodes (mixer, concat), use targetHandle to distinguish
      if (e.targetHandle === 'audio_1' && src.audioBase64) merged.audioBase64 = src.audioBase64
      if (e.targetHandle === 'audio_2' && src.audioBase64) merged.text = src.audioBase64 // store second track in text field temporarily
      if (e.targetHandle === 'speech' && src.audioBase64) merged.audioBase64 = src.audioBase64
      if (e.targetHandle === 'bgm' && src.audioBase64) merged.text = src.audioBase64
    }
  }
  return merged
}

const nodeExecutors: Record<NodeType, (node: WorkflowNode, inputs: NodeOutput) => Promise<NodeOutput>> = {
  text_input: async (node) => ({
    text: (node.data.text as string) ?? '',
  }),

  ref_audio: async (node) => ({
    audioBase64: (node.data.audioBase64 as string) ?? '',
    sampleRate: (node.data.sampleRate as number) ?? 24000,
    audioPath: (node.data.path as string) ?? '',
  }),

  tts_engine: async (node, inputs) => {
    const text = inputs.text ?? ''
    if (!text.trim()) throw new Error('TTS 节点缺少文本输入')

    const engine = (node.data.engine as string) ?? 'cosyvoice2'
    const voice = (node.data.voice as string) ?? 'default'
    const speed = (node.data.speed as number) ?? 1.0
    const sampleRate = (node.data.sampleRate as number) ?? 24000
    const emotion = (node.data.emotion as string) || undefined

    const resp = await apiFetch<SynthesizeResponse>('/api/v1/tts/synthesize', {
      method: 'POST',
      body: JSON.stringify({
        engine, text, voice, speed, sample_rate: sampleRate,
        reference_audio: inputs.audioPath ?? undefined,
        emotion,
      }),
    })

    return {
      audioBase64: resp.audio_base64,
      sampleRate: resp.sample_rate,
    }
  },

  resample: async (_node, inputs) => {
    // TODO: call nous-core /audio/resample when available
    // For now, passthrough
    return { audioBase64: inputs.audioBase64, sampleRate: inputs.sampleRate }
  },

  mixer: async (_node, inputs) => {
    // TODO: WASM audio mixing
    return { audioBase64: inputs.audioBase64, sampleRate: inputs.sampleRate }
  },

  concat: async (_node, inputs) => {
    // TODO: call nous-core /audio/concat or WASM
    return { audioBase64: inputs.audioBase64, sampleRate: inputs.sampleRate }
  },

  bgm_mix: async (_node, inputs) => {
    // TODO: WASM BGM mixing
    return { audioBase64: inputs.audioBase64, sampleRate: inputs.sampleRate }
  },

  output: async (_node, inputs) => {
    return { audioBase64: inputs.audioBase64, sampleRate: inputs.sampleRate }
  },
}

/**
 * Check if workflow contains plugin nodes (not in built-in executors).
 * If so, execute on backend instead of frontend.
 */
function hasPluginNodes(nodes: WorkflowNode[]): boolean {
  return nodes.some((n) => !(n.type in nodeExecutors))
}

/**
 * Execute workflow on backend via API.
 * Used for workflows containing plugin nodes.
 */
async function executeOnBackend(workflow: Workflow): Promise<ExecutionResult> {
  const result = await apiFetch<{ outputs: Record<string, Record<string, unknown>> }>(
    '/api/v1/workflows/execute',
    {
      method: 'POST',
      body: JSON.stringify({ nodes: workflow.nodes, edges: workflow.edges }),
    }
  )

  // Find output node result
  const outputNodeId = workflow.nodes.find((n) => n.type === 'output')?.id
  if (!outputNodeId) throw new Error('工作流缺少输出节点')

  const outputData = result.outputs[outputNodeId]
  const audio = (outputData?.audio as string) ?? (outputData?.audioBase64 as string) ?? ''

  if (!audio) {
    // Maybe it's text-only output
    const text = outputData?.text as string
    if (text) {
      return { audioBase64: '', sampleRate: 24000, duration: 0 }
    }
    throw new Error('工作流执行完成但没有音频输出')
  }

  return {
    audioBase64: audio,
    sampleRate: (outputData?.sample_rate as number) ?? (outputData?.sampleRate as number) ?? 24000,
    duration: 0,
  }
}

export async function executeWorkflow(workflow: Workflow): Promise<ExecutionResult> {
  const { nodes, edges } = workflow

  if (nodes.length === 0) throw new Error('工作流为空')

  const hasOutput = nodes.some((n) => n.type === 'output')
  if (!hasOutput) throw new Error('工作流缺少输出节点')

  // If workflow has plugin nodes, execute on backend
  if (hasPluginNodes(nodes)) {
    return executeOnBackend(workflow)
  }

  const sorted = topoSort(nodes, edges)
  const outputs = new Map<string, NodeOutput>()

  for (const node of sorted) {
    const inputs = getInputs(node.id, edges, outputs)
    const executor = nodeExecutors[node.type]
    if (!executor) throw new Error(`未知节点类型: ${node.type}`)

    const result = await executor(node, inputs)
    outputs.set(node.id, result)
  }

  // Find the output node's result
  const outputNode = sorted.find((n) => n.type === 'output')!
  const finalOutput = outputs.get(outputNode.id)

  if (!finalOutput?.audioBase64) {
    throw new Error('工作流执行完成但没有音频输出')
  }

  return {
    audioBase64: finalOutput.audioBase64,
    sampleRate: finalOutput.sampleRate ?? 24000,
    duration: 0, // will be calculated by player
  }
}
