import { useState, useEffect, useMemo } from 'react'
import { Download, Maximize2, ImageOff } from 'lucide-react'
import { NodeResizer, type NodeProps } from '@xyflow/react'
import { NODE_DEFS } from '../../models/workflow'
import { useWorkspaceStore } from '../../stores/workspace'
import { useLightboxStore } from '../../stores/lightbox'
import BaseNode from './BaseNode'

type Phase = 'empty' | 'loading' | 'success' | 'error'

export default function ImageOutputNode({ id, data, selected }: NodeProps) {
  const def = NODE_DEFS.image_output
  const tabs = useWorkspaceStore((s) => s.tabs)
  const activeTabId = useWorkspaceStore((s) => s.activeTabId)
  const updateNode = useWorkspaceStore((s) => s.updateNode)
  const upstreamNodeId = useMemo(() => {
    const wf = tabs.find((t) => t.id === activeTabId)?.workflow
    const edge = wf?.edges.find((e) => e.target === id)
    return edge?.source ?? null
  }, [tabs, activeTabId, id])

  // 对齐 IC「生成图 vs 输入图」对比:沿 edges 上溯,找喂进本输出的源图 url
  // (编辑/超分流的 image_input/源节点)。纯文生图无源图 → null,灯箱对比按钮不出现。
  // 中间采样/编码节点不带 image url,所以第一个命中的就是真正的输入源。
  const compareBaseUrl = useMemo(() => {
    const wf = tabs.find((t) => t.id === activeTabId)?.workflow
    if (!wf) return null
    const incoming = new Map<string, string[]>()
    for (const e of wf.edges) {
      if (!incoming.has(e.target)) incoming.set(e.target, [])
      incoming.get(e.target)!.push(e.source)
    }
    const nodeById = new Map(wf.nodes.map((n) => [n.id, n]))
    const IMG_KEYS = ['image_url', 'image', 'url']
    const seen = new Set<string>([id])
    const queue = [...(incoming.get(id) ?? [])]
    while (queue.length) {
      const cur = queue.shift()!
      if (seen.has(cur)) continue
      seen.add(cur)
      const d = nodeById.get(cur)?.data as Record<string, unknown> | undefined
      for (const k of IMG_KEYS) {
        const v = d?.[k]
        if (typeof v === 'string' && v) return v
      }
      for (const s of incoming.get(cur) ?? []) queue.push(s)
    }
    return null
  }, [tabs, activeTabId, id])

  const [phase, setPhase] = useState<Phase>('empty')
  const [error, setError] = useState<string>('')
  const openLightbox = useLightboxStore((s) => s.openFromUrl)

  const imageUrl = (data.image_url as string) || ''
  const mediaType = (data.media_type as string) || 'image/png'
  const dataUrl = imageUrl
  const seed = data.seed as number | null | undefined
  const steps = data.steps as number | null | undefined
  const cfgScale = data.cfg_scale as number | null | undefined
  const width = data.width as number | null | undefined
  const height = data.height as number | null | undefined
  const durationMs = data.duration_ms as number | null | undefined

  const captionParts: string[] = []
  if (seed !== null && seed !== undefined) captionParts.push(`seed: ${seed}`)
  if (steps !== null && steps !== undefined) captionParts.push(`${steps} steps`)
  if (cfgScale !== null && cfgScale !== undefined) captionParts.push(`cfg ${cfgScale}`)
  if (width && height) captionParts.push(`${width}×${height}`)
  if (durationMs !== null && durationMs !== undefined) captionParts.push(`${(durationMs / 1000).toFixed(1)}s`)

  // 灯箱元信息(画布输出节点:无 prompt,展 seed/steps/cfg/分辨率/时长)。
  const lbFields: Array<{ label: string; value: string }> = []
  if (seed !== null && seed !== undefined) lbFields.push({ label: 'seed', value: String(seed) })
  if (steps !== null && steps !== undefined) lbFields.push({ label: 'steps', value: String(steps) })
  if (cfgScale !== null && cfgScale !== undefined) lbFields.push({ label: 'cfg', value: String(cfgScale) })
  const openZoom = () => {
    if (!dataUrl) return
    openLightbox(dataUrl, {
      resolution: width && height ? `${width}×${height}` : undefined,
      durationMs: durationMs ?? null,
      fields: lbFields,
      compareBase: compareBaseUrl ?? undefined,
    })
  }

  useEffect(() => {
    if (imageUrl && phase !== 'success') setPhase('success')
  }, [imageUrl, phase])

  useEffect(() => {
    const handler = (event: Event) => {
      const detail = (event as CustomEvent).detail
      const src = upstreamNodeId
      if (detail.type === 'node_start' && src && detail.node_id === src) {
        setPhase('loading')
        setError('')
      }
      if (detail.type === 'node_error' && src && detail.node_id === src) {
        setPhase('error')
        setError(typeof detail.error === 'string' ? detail.error : '生成失败')
      }
      // Lane S 异步:图像结果经 node_complete 带回(后端 workflow_executor 把 image_url 等
      // 塞进 node_complete)。上游 VAE Decode 或本输出节点带 image_url → 写回 + 显示预览。
      if (
        detail.type === 'node_complete' &&
        detail.image_url &&
        (detail.node_id === id || detail.node_id === src)
      ) {
        updateNode(id, {
          image_url: detail.image_url,
          media_type: detail.media_type ?? 'image/png',
          width: detail.width ?? null,
          height: detail.height ?? null,
          seed: detail.seed ?? null,
          steps: detail.steps ?? null,
          cfg_scale: detail.cfg_scale ?? null,
          duration_ms: detail.duration_ms ?? null,
        })
        setPhase('success')
        setError('')
      }
    }
    window.addEventListener('node-progress', handler)
    return () => window.removeEventListener('node-progress', handler)
  }, [upstreamNodeId, id, updateNode])

  const onDownload = () => {
    if (!dataUrl) return
    const a = document.createElement('a')
    a.href = dataUrl
    a.download = `image-${Date.now()}.${mediaType.split('/')[1] || 'png'}`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  return (
    <>
      <NodeResizer
        isVisible={selected}
        minWidth={280}
        minHeight={320}
        onResizeEnd={() => window.dispatchEvent(new Event('node-resize-end'))}
        lineStyle={{ border: 'none' }}
        handleStyle={{ width: 12, height: 12, background: 'transparent', border: 'none' }}
      />
      <BaseNode
        title={def.label}
        badge={{ label: 'IMG', bg: 'rgba(59,130,246,0.15)', color: 'var(--info)' }}
        selected={selected}
        inputs={def.inputs}
        outputs={def.outputs}
      >
        <div style={{ padding: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div
            style={{
              width: '100%',
              aspectRatio: '1 / 1',
              background: 'var(--bg)',
              borderRadius: 4,
              border: '1px solid var(--border)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              overflow: 'hidden',
              position: 'relative',
              cursor: phase === 'success' ? 'zoom-in' : 'default',
            }}
            onClick={() => phase === 'success' && openZoom()}
          >
            {phase === 'success' && dataUrl ? (
              <img
                src={dataUrl}
                alt="generated"
                style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
              />
            ) : phase === 'loading' ? (
              <div style={{ fontSize: 11, color: 'var(--muted)' }}>生成中...</div>
            ) : phase === 'error' ? (
              <div
                style={{
                  fontSize: 10,
                  color: 'var(--err)',
                  textAlign: 'center',
                  padding: 8,
                  whiteSpace: 'pre-wrap',
                }}
              >
                {error || '生成失败'}
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, color: 'var(--muted-strong)' }}>
                <ImageOff size={20} />
                <span style={{ fontSize: 10 }}>等待生成</span>
              </div>
            )}
          </div>
          {phase === 'success' && (
            <div style={{ display: 'flex', gap: 4 }}>
              <button
                className="nodrag"
                onClick={onDownload}
                style={{
                  flex: 1,
                  padding: '4px 6px',
                  fontSize: 10,
                  background: 'var(--bg-hover)',
                  border: '1px solid var(--border)',
                  borderRadius: 3,
                  cursor: 'pointer',
                  color: 'var(--text)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 4,
                }}
              >
                <Download size={10} />
                下载
              </button>
              <button
                className="nodrag"
                onClick={openZoom}
                style={{
                  flex: 1,
                  padding: '4px 6px',
                  fontSize: 10,
                  background: 'var(--bg-hover)',
                  border: '1px solid var(--border)',
                  borderRadius: 3,
                  cursor: 'pointer',
                  color: 'var(--text)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 4,
                }}
              >
                <Maximize2 size={10} />
                放大
              </button>
            </div>
          )}
          {phase === 'success' && captionParts.length > 0 && (
            <div
              style={{
                fontSize: 9,
                color: 'var(--muted)',
                lineHeight: 1.4,
                wordBreak: 'break-word',
              }}
            >
              {captionParts.join(' · ')}
            </div>
          )}
        </div>
      </BaseNode>
    </>
  )
}
