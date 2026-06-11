import { useState, useEffect, useMemo } from 'react'
import { Download, Maximize2, ImageOff, X, Trash2, DownloadCloud } from 'lucide-react'
import { NodeResizer, type NodeProps } from '@xyflow/react'
import { NODE_DEFS } from '../../models/workflow'
import { useWorkspaceStore } from '../../stores/workspace'
import { useLightboxStore, type LightboxMeta } from '../../stores/lightbox'
import { appendOutput, type OutImg } from './imageOutputGallery'
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

  // 仅持有「活动态」(运行中/失败);success/empty 从 images.length 派生 —— 避免 setState-in-effect。
  const [act, setAct] = useState<'idle' | 'loading' | 'error'>('idle')
  const [error, setError] = useState<string>('')
  const openItems = useLightboxStore((s) => s.openItems)

  // 画廊:优先读累积 images[];兼容旧单图(只存 image_url 顶层)→ 包成 1 项。
  const images = useMemo<OutImg[]>(() => {
    const arr = Array.isArray(data.images) ? (data.images as OutImg[]) : []
    if (arr.length) return arr
    if (data.image_url) {
      return [{
        url: data.image_url as string,
        seed: (data.seed as number) ?? null,
        steps: (data.steps as number) ?? null,
        cfg: (data.cfg_scale as number) ?? null,
        width: (data.width as number) ?? null,
        height: (data.height as number) ?? null,
        durationMs: (data.duration_ms as number) ?? null,
      }]
    }
    return []
  }, [data.images, data.image_url, data.seed, data.steps, data.cfg_scale, data.width, data.height, data.duration_ms])

  const mediaType = (data.media_type as string) || 'image/png'

  // 派生显示态:运行中/失败优先,否则有图=success / 无图=empty(含恢复已存图的工作流,无需事件)。
  const phase: Phase = act === 'loading' ? 'loading'
    : act === 'error' ? 'error'
      : images.length ? 'success' : 'empty'

  useEffect(() => {
    const handler = (event: Event) => {
      const detail = (event as CustomEvent).detail
      const src = upstreamNodeId
      if (detail.type === 'node_start' && src && detail.node_id === src) {
        setAct('loading')
        setError('')
      }
      if (detail.type === 'node_error' && src && detail.node_id === src) {
        setAct('error')
        setError(typeof detail.error === 'string' ? detail.error : '生成失败')
      }
      // Lane S 异步:图像结果经 node_complete 带回。**累积**(对齐 IC):每次运行 append 一张,
      // 不覆盖。读 store 当前 images 再追加(handler 闭包里的 data 是旧值)。dedup by url。
      if (
        detail.type === 'node_complete' &&
        detail.image_url &&
        (detail.node_id === id || detail.node_id === src)
      ) {
        const cur = useWorkspaceStore.getState().getActiveWorkflow().nodes.find((n) => n.id === id)?.data
        const prev: OutImg[] = Array.isArray(cur?.images) ? (cur!.images as OutImg[]) : []
        // batch(PR-B1):node_complete 可能带 image_urls 列表(一次 N 张);否则单 image_url。
        // 同次 batch 共享 seed/steps/cfg 等元信息(node_complete 只回一份)。逐张 append(dedup)。
        const urls: string[] = Array.isArray(detail.image_urls) && detail.image_urls.length
          ? (detail.image_urls as string[])
          : [detail.image_url as string]
        const base = {
          seed: detail.seed ?? null, steps: detail.steps ?? null, cfg: detail.cfg_scale ?? null,
          width: detail.width ?? null, height: detail.height ?? null, durationMs: detail.duration_ms ?? null,
        }
        let next = prev
        for (const u of urls) next = appendOutput(next, { url: u, ...base })
        if (next !== prev) {
          updateNode(id, {
            images: next,
            // 顶层镜像「最新」一张(灯箱 collectWorkflowImages / compareBase 兜底读 image_url)。
            image_url: urls[urls.length - 1],
            media_type: detail.media_type ?? 'image/png',
            width: base.width, height: base.height, seed: base.seed,
            steps: base.steps, cfg_scale: base.cfg, duration_ms: base.durationMs,
          })
        }
        setAct('idle')  // 清运行态;有图 → 派生 phase=success
        setError('')
      }
    }
    window.addEventListener('node-progress', handler)
    return () => window.removeEventListener('node-progress', handler)
  }, [upstreamNodeId, id, updateNode])

  const metaFor = (im: OutImg): LightboxMeta => {
    const fields: Array<{ label: string; value: string }> = []
    if (im.seed !== null && im.seed !== undefined) fields.push({ label: 'seed', value: String(im.seed) })
    if (im.steps !== null && im.steps !== undefined) fields.push({ label: 'steps', value: String(im.steps) })
    if (im.cfg !== null && im.cfg !== undefined) fields.push({ label: 'cfg', value: String(im.cfg) })
    return {
      resolution: im.width && im.height ? `${im.width}×${im.height}` : undefined,
      durationMs: im.durationMs ?? null,
      fields,
      compareBase: compareBaseUrl ?? undefined,
    }
  }
  const openAt = (i: number) => {
    if (!images.length) return
    openItems(images.map((im) => ({ url: im.url, meta: metaFor(im) })), i)
  }

  const basename = (u: string): string => {
    try {
      const path = new URL(u, window.location.href).pathname
      return decodeURIComponent(path.split('/').pop() || '') || `image.${mediaType.split('/')[1] || 'png'}`
    } catch { return `image-${Date.now()}.${mediaType.split('/')[1] || 'png'}` }
  }
  const downloadOne = (u: string) => {
    const a = document.createElement('a')
    a.href = u; a.download = basename(u)
    document.body.appendChild(a); a.click(); a.remove()
  }
  const downloadAll = () => images.forEach((im, i) => setTimeout(() => downloadOne(im.url), i * 200))

  const removeAt = (i: number) => {
    const next = images.filter((_, j) => j !== i)
    updateNode(id, { images: next, image_url: next.length ? next[next.length - 1].url : '' })
    // 空了 → 派生 phase=empty,无需手动置态
  }
  const clearAll = () => updateNode(id, { images: [], image_url: '' })

  const single = images.length === 1
  const multi = images.length > 1

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
        badge={multi
          ? { label: `${images.length} 张`, bg: 'rgba(59,130,246,0.15)', color: 'var(--info)' }
          : { label: 'IMG', bg: 'rgba(59,130,246,0.15)', color: 'var(--info)' }}
        selected={selected}
        inputs={def.inputs}
        outputs={def.outputs}
      >
        <div style={{ padding: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
          {phase === 'success' && multi ? (
            // 多图:网格画廊(对齐 IC OUTPUT)。每格点开灯箱 + hover 删除。
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(72px, 1fr))',
                gap: 4,
                maxHeight: 360,
                overflowY: 'auto',
              }}
            >
              {images.map((im, i) => (
                <div
                  key={im.url}
                  className="nodrag"
                  onClick={() => openAt(i)}
                  style={{
                    position: 'relative', aspectRatio: '1 / 1', borderRadius: 3, overflow: 'hidden',
                    background: 'var(--bg)', border: '1px solid var(--border)', cursor: 'zoom-in',
                  }}
                >
                  <img src={im.url} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                  <button
                    className="nodrag"
                    onClick={(e) => { e.stopPropagation(); removeAt(i) }}
                    title="删除此图"
                    style={{
                      position: 'absolute', top: 2, right: 2, width: 16, height: 16, borderRadius: 3,
                      border: 'none', background: 'rgba(0,0,0,0.55)', color: '#fff', cursor: 'pointer',
                      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 0,
                    }}
                  >
                    <X size={10} />
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div
              className={phase === 'success' && single ? 'nodrag' : undefined}
              style={{
                width: '100%', aspectRatio: '1 / 1', background: 'var(--bg)', borderRadius: 4,
                border: '1px solid var(--border)', display: 'flex', alignItems: 'center',
                justifyContent: 'center', overflow: 'hidden', position: 'relative',
                cursor: phase === 'success' && single ? 'zoom-in' : 'default',
              }}
              onClick={() => phase === 'success' && single && openAt(0)}
            >
              {phase === 'success' && single ? (
                <img src={images[0].url} alt="generated" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
              ) : phase === 'loading' ? (
                <div style={{ fontSize: 11, color: 'var(--muted)' }}>生成中...</div>
              ) : phase === 'error' ? (
                <div style={{ fontSize: 10, color: 'var(--err)', textAlign: 'center', padding: 8, whiteSpace: 'pre-wrap' }}>
                  {error || '生成失败'}
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, color: 'var(--muted-strong)' }}>
                  <ImageOff size={20} />
                  <span style={{ fontSize: 10 }}>等待生成</span>
                </div>
              )}
            </div>
          )}
          {phase === 'success' && images.length > 0 && (
            <div style={{ display: 'flex', gap: 4 }}>
              {single ? (
                <>
                  <NodeBtn onClick={() => downloadOne(images[0].url)} icon={<Download size={10} />} label="下载" />
                  <NodeBtn onClick={() => openAt(0)} icon={<Maximize2 size={10} />} label="放大" />
                </>
              ) : (
                <>
                  <NodeBtn onClick={downloadAll} icon={<DownloadCloud size={10} />} label={`下载全部(${images.length})`} />
                  <NodeBtn onClick={clearAll} icon={<Trash2 size={10} />} label="清空" />
                </>
              )}
            </div>
          )}
        </div>
      </BaseNode>
    </>
  )
}

function NodeBtn({ onClick, icon, label }: { onClick: () => void; icon: React.ReactNode; label: string }) {
  return (
    <button
      className="nodrag"
      onClick={onClick}
      style={{
        flex: 1, padding: '4px 6px', fontSize: 10, background: 'var(--bg-hover)',
        border: '1px solid var(--border)', borderRadius: 3, cursor: 'pointer', color: 'var(--text)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4,
      }}
    >
      {icon}
      {label}
    </button>
  )
}
