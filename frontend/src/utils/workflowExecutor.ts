import type { Workflow, WorkflowNode, WorkflowEdge, NodeType } from '../models/workflow'
import { apiFetch } from '../api/client'
import type { SynthesizeResponse } from '../api/tts'
import { useExecutionStore } from '../stores/execution'
import { useWorkspaceStore } from '../stores/workspace'

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
  // Mark all nodes pending so the UI has visible state before the first
  // node_start event lands.
  const exec = useExecutionStore.getState()
  for (const n of workflow.nodes) exec.setNodeState(n.id, 'pending')

  // Open a progress channel. Server pushes node_start/complete/error into
  // this bucket; ws must be connected BEFORE the POST so events aren't lost.
  const channelId = `ch-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
  const ws = await openProgressChannel(channelId)

  try {
    const result = await apiFetch<{ outputs: Record<string, Record<string, unknown>> }>(
      '/api/v1/workflows/execute',
      {
        method: 'POST',
        body: JSON.stringify({
          nodes: workflow.nodes,
          edges: workflow.edges,
          name: workflow.name,
          channel_id: channelId,
        }),
      }
    )
    return unwrapOutputs(result, workflow)
  } finally {
    ws.close()
  }
}


function openProgressChannel(channelId: string): Promise<WebSocket> {
  return new Promise((resolve, reject) => {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${proto}//${window.location.host}/ws/workflow/${channelId}`)
    const exec = useExecutionStore.getState()
    ws.onopen = () => resolve(ws)
    ws.onerror = (e) => reject(new Error(`ws open failed: ${String(e)}`))
    ws.onmessage = (ev) => {
      try {
        const d = JSON.parse(ev.data)
        // Dispatch window CustomEvent so DeclarativeNode / TextOutputNode
        // can pick up node_stream / node_complete for streaming text + stats.
        window.dispatchEvent(new CustomEvent('node-progress', { detail: d }))

        if (d.type === 'node_start') {
          exec.setNodeState(d.node_id, 'running')
          exec.setCurrentNode(d.node_id, d.node_type ?? null)
          if (typeof d.progress === 'number') exec.setProgress(d.progress)
        } else if (d.type === 'node_complete') {
          exec.clearNodeState(d.node_id)
          if (typeof d.progress === 'number') exec.setProgress(d.progress)
        } else if (d.type === 'node_error') {
          exec.setNodeState(d.node_id, 'error')
        } else if (d.type === 'complete') {
          exec.setProgress(100)
          exec.setCurrentNode(null, null)
        }
      } catch { /* ignore parse errors */ }
    }
  })
}


function unwrapOutputs(
  result: { outputs: Record<string, Record<string, unknown>> },
  workflow: Workflow,
): ExecutionResult {

  // Find output node result — support both audio output and text output
  const audioOutputNode = workflow.nodes.find((n) => n.type === 'output')
  const textOutputNode = workflow.nodes.find((n) => n.type === 'text_output')

  if (audioOutputNode) {
    const outputData = result.outputs[audioOutputNode.id]
    const audio = (outputData?.audio as string) ?? (outputData?.audioBase64 as string) ?? ''
    if (audio) {
      return {
        audioBase64: audio,
        sampleRate: (outputData?.sample_rate as number) ?? (outputData?.sampleRate as number) ?? 24000,
        duration: 0,
      }
    }
  }

  if (textOutputNode) {
    const outputData = result.outputs[textOutputNode.id]
    const text = (outputData?.text as string) ?? ''
    // Update the text_output node's data to display the result
    const { updateNode } = useWorkspaceStore.getState()
    updateNode(textOutputNode.id, { text })
    return { audioBase64: '', sampleRate: 24000, duration: 0 }
  }

  throw new Error('工作流执行完成但没有输出')
}

async function recordTask(data: {
  workflow_name: string
  status: string
  nodes_total: number
  nodes_done: number
  duration_ms: number
  error?: string
}) {
  try {
    await apiFetch('/api/v1/tasks/record', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  } catch {
    // ignore if recording fails
  }
}

export async function executeWorkflow(workflow: Workflow): Promise<ExecutionResult> {
  const { nodes, edges } = workflow

  if (nodes.length === 0) throw new Error('工作流为空')

  const hasOutput = nodes.some((n) => n.type === 'output' || n.type === 'text_output')
  if (!hasOutput) throw new Error('工作流缺少输出节点')

  // If workflow has plugin nodes, execute on backend (task record is created server-side)
  if (hasPluginNodes(nodes)) {
    return executeOnBackend(workflow)
  }

  const sorted = topoSort(nodes, edges)
  const outputs = new Map<string, NodeOutput>()
  const exec = useExecutionStore.getState()
  const startTime = performance.now()

  // Mark all nodes as pending
  for (const node of sorted) {
    exec.setNodeState(node.id, 'pending')
  }

  for (let i = 0; i < sorted.length; i++) {
    const node = sorted[i]
    const inputs = getInputs(node.id, edges, outputs)
    const executor = nodeExecutors[node.type]
    if (!executor) throw new Error(`未知节点类型: ${node.type}`)

    exec.setNodeState(node.id, 'running')
    exec.setCurrentNode(node.id, node.type)
    exec.setProgress(Math.round(((i) / sorted.length) * 100))

    try {
      const result = await executor(node, inputs)
      outputs.set(node.id, result)
      // Clear once done — only the currently-running node stays highlighted.
      exec.clearNodeState(node.id)
    } catch (e) {
      exec.setNodeState(node.id, 'error')
      const elapsed = Math.round(performance.now() - startTime)
      recordTask({
        workflow_name: workflow.name || '前端执行',
        status: 'failed',
        nodes_total: sorted.length,
        nodes_done: i,
        duration_ms: elapsed,
        error: e instanceof Error ? e.message : String(e),
      })
      throw e
    }
  }

  // Find the output node's result
  const outputNode = sorted.find((n) => n.type === 'output')!
  const finalOutput = outputs.get(outputNode.id)

  const elapsed = Math.round(performance.now() - startTime)

  if (!finalOutput?.audioBase64) {
    recordTask({
      workflow_name: workflow.name || '前端执行',
      status: 'failed',
      nodes_total: sorted.length,
      nodes_done: sorted.length,
      duration_ms: elapsed,
      error: '工作流执行完成但没有音频输出',
    })
    throw new Error('工作流执行完成但没有音频输出')
  }

  // Record successful task
  recordTask({
    workflow_name: workflow.name || '前端执行',
    status: 'completed',
    nodes_total: sorted.length,
    nodes_done: sorted.length,
    duration_ms: elapsed,
  })

  return {
    audioBase64: finalOutput.audioBase64,
    sampleRate: finalOutput.sampleRate ?? 24000,
    duration: 0, // will be calculated by player
  }
}
