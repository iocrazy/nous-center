import { useEffect, useMemo, useRef, useState } from 'react'
import { ImageIcon, Sparkles, Wand2, Scan, Box, Loader2, Monitor, Cloud } from 'lucide-react'
import { useComponents } from '../api/components'
import { executeWorkflow } from '../utils/workflowExecutor'
import type { Workflow } from '../models/workflow'
import { useToastStore } from '../stores/toast'

// 创作台:对齐 Infinite-Canvas 的「统一创作控制台」,但引擎是 nous 本地(不外接 ComfyUI)。
// 客户端搭 z-image 工作流图 → /api/v1/workflows/execute(admin cookie 鉴权)→ WS 拿 image_url。
// 四功能:文生图(Z-Image,已通)/ 细节增强 / 图片编辑 / 角度控制(后三个随 P2/P3 引擎上线接)。

type FeatureId = 'text2img' | 'enhance' | 'edit' | 'angle'

const FEATURES: { id: FeatureId; label: string; icon: typeof ImageIcon; ready: boolean }[] = [
  { id: 'text2img', label: '文生图', icon: ImageIcon, ready: true },
  { id: 'enhance', label: '细节增强', icon: Sparkles, ready: false },
  { id: 'edit', label: '图片编辑', icon: Wand2, ready: false },
  { id: 'angle', label: '角度控制', icon: Box, ready: false },
]

interface GalleryItem { url: string; prompt: string; seed: number }

export default function Studio() {
  const [feature, setFeature] = useState<FeatureId>('text2img')
  return (
    <div style={{ position: 'absolute', inset: 0, display: 'flex', background: 'var(--bg)', overflow: 'hidden' }}>
      {/* 左侧功能子导航(对齐 Infinite-Canvas 侧栏) */}
      <div style={{
        width: 168, flexShrink: 0, borderRight: '1px solid var(--border)',
        padding: '16px 10px', background: 'var(--bg-accent)',
      }}>
        <div style={{ fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: 0.6, padding: '0 8px 8px' }}>
          本地功能
        </div>
        {FEATURES.map(({ id, label, icon: Icon, ready }) => {
          const active = feature === id
          return (
            <button
              key={id}
              type="button"
              onClick={() => ready && setFeature(id)}
              disabled={!ready}
              style={{
                display: 'flex', alignItems: 'center', gap: 9, width: '100%',
                padding: '9px 10px', marginBottom: 3, borderRadius: 8, fontSize: 13,
                border: 'none', textAlign: 'left',
                background: active ? 'var(--text)' : 'transparent',
                color: active ? 'var(--bg)' : ready ? 'var(--text)' : 'var(--muted)',
                cursor: ready ? 'pointer' : 'not-allowed', opacity: ready ? 1 : 0.55,
                fontWeight: active ? 600 : 400,
              }}
            >
              <Icon size={16} />
              <span style={{ flex: 1 }}>{label}</span>
              {!ready && <span style={{ fontSize: 9, color: 'var(--muted)' }}>待接</span>}
            </button>
          )
        })}
      </div>

      {/* 主区 */}
      <div style={{ flex: 1, overflow: 'auto' }}>
        {feature === 'text2img' ? (
          <Text2ImagePanel />
        ) : (
          <ComingSoon label={FEATURES.find((f) => f.id === feature)?.label ?? ''} />
        )}
      </div>
    </div>
  )
}

function ComingSoon({ label }: { label: string }) {
  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 10, color: 'var(--muted)' }}>
      <Scan size={28} />
      <div style={{ fontSize: 14 }}>「{label}」引擎接入中</div>
      <div style={{ fontSize: 12 }}>P2/P3 上线后这里就能用本地引擎跑</div>
    </div>
  )
}

function Text2ImagePanel() {
  const toast = useToastStore((s) => s.add)
  const { data: checkpoints } = useComponents('checkpoint')
  const [prompt, setPrompt] = useState('')
  const [width, setWidth] = useState(1024)
  const [height, setHeight] = useState(1024)
  const [ckpt, setCkpt] = useState<string>('')
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const [gallery, setGallery] = useState<GalleryItem[]>([])
  const lastSeed = useRef(0)

  // 默认挑一个像 Z-Image 的整模型(否则用第一个)。
  const zCandidates = useMemo(
    () => (checkpoints ?? []).filter((c) => /z[-_ ]?image/i.test(c.filename) || /z[-_ ]?image/i.test(c.abs_path)),
    [checkpoints],
  )
  useEffect(() => {
    if (ckpt) return
    const pick = zCandidates[0] ?? (checkpoints ?? [])[0]
    if (pick) setCkpt(pick.abs_path)
  }, [ckpt, zCandidates, checkpoints])

  const run = async () => {
    if (running) return
    if (!prompt.trim()) { toast('先描述你想生成的画面', 'info'); return }
    if (!ckpt) { toast('没找到可用的 Z-Image 整模型(在 diffusers/ 放 Z-Image-Turbo)', 'error'); return }
    setRunning(true)
    setResult(null)
    const seed = Math.floor(Math.random() * 1_000_000_000_000)
    lastSeed.current = seed
    const wf = buildZImageWorkflow({ ckpt, prompt: prompt.trim(), width, height, seed })

    let settled = false
    const onProgress = (ev: Event) => {
      const d = (ev as CustomEvent).detail
      if (settled) return
      if (d?.type === 'node_complete' && d?.image_url) {
        settled = true
        const url = String(d.image_url)
        setResult(url)
        setGallery((g) => [{ url, prompt: prompt.trim(), seed }, ...g].slice(0, 24))
        setRunning(false)
      } else if (d?.type === 'node_error') {
        settled = true
        toast(`生成失败:${d.error ?? d.detail ?? '未知错误'}`, 'error')
        setRunning(false)
      }
    }
    window.addEventListener('node-progress', onProgress as EventListener)
    try {
      await executeWorkflow(wf)
      toast('已入队本地生成…', 'info')
    } catch (err) {
      settled = true
      toast(`提交失败:${(err as Error).message ?? err}`, 'error')
      setRunning(false)
    }
    // 兜底:90s 没出图 → 解绑(防泄漏);真结果靠 WS。
    setTimeout(() => {
      window.removeEventListener('node-progress', onProgress as EventListener)
      if (!settled) setRunning(false)
    }, 90_000)
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 24 }}>
      {/* 控制台卡片 */}
      <div style={{ background: 'var(--bg-accent)', border: '1px solid var(--border)', borderRadius: 12, padding: 18 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
          <span style={{ fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: 1 }}>
            统一创作控制台 · 本地引擎
          </span>
          <span style={{ fontSize: 11, color: 'var(--ok)', display: 'inline-flex', alignItems: 'center', gap: 5 }}>
            系统就绪 <span style={{ width: 6, height: 6, borderRadius: 3, background: 'var(--ok)' }} />
          </span>
        </div>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="描述你想生成的画面…"
          rows={3}
          style={{
            width: '100%', resize: 'vertical', background: 'var(--bg)', color: 'var(--text)',
            border: '1px solid var(--border)', borderRadius: 8, padding: '12px 14px',
            fontSize: 15, lineHeight: 1.5, outline: 'none',
          }}
        />
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: 18, marginTop: 14, flexWrap: 'wrap' }}>
          {/* 引擎来源 */}
          <Field label="引擎来源">
            <div style={{ display: 'flex', border: '1px solid var(--border)', borderRadius: 8, overflow: 'hidden' }}>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, padding: '7px 12px', fontSize: 12, background: 'var(--text)', color: 'var(--bg)' }}>
                <Monitor size={13} /> 本地
              </span>
              <span title="云端来源后续接入" style={{ display: 'inline-flex', alignItems: 'center', gap: 5, padding: '7px 12px', fontSize: 12, color: 'var(--muted)' }}>
                <Cloud size={13} /> ModelScope
              </span>
            </div>
          </Field>
          {/* 模型 */}
          <Field label="模型(整模型)">
            <select
              value={ckpt}
              onChange={(e) => setCkpt(e.target.value)}
              style={selectStyle}
            >
              {(checkpoints ?? []).length === 0 && <option value="">无可用整模型</option>}
              {(checkpoints ?? []).map((c) => (
                <option key={c.abs_path} value={c.abs_path}>{c.filename}</option>
              ))}
            </select>
          </Field>
          {/* 尺寸 */}
          <Field label="尺寸">
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <NumBox value={width} onChange={setWidth} />
              <span style={{ color: 'var(--muted)' }}>×</span>
              <NumBox value={height} onChange={setHeight} />
            </div>
          </Field>
          <div style={{ flex: 1 }} />
          <button
            type="button"
            onClick={run}
            disabled={running}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 8, padding: '11px 22px',
              background: 'var(--text)', color: 'var(--bg)', border: 'none', borderRadius: 8,
              fontSize: 14, fontWeight: 600, cursor: running ? 'wait' : 'pointer', opacity: running ? 0.7 : 1,
            }}
          >
            {running ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
            {running ? '本地生成中…' : '本地生成'}
          </button>
        </div>
      </div>

      {/* 当前结果 */}
      {(running || result) && (
        <div style={{ marginTop: 18, display: 'flex', justifyContent: 'center' }}>
          <div style={{
            width: 420, height: 420, borderRadius: 12, border: '1px solid var(--border)',
            background: 'var(--bg-accent)', display: 'flex', alignItems: 'center', justifyContent: 'center',
            overflow: 'hidden',
          }}>
            {result ? (
              <img src={result} alt="生成结果" style={{ maxWidth: '100%', maxHeight: '100%', display: 'block' }} />
            ) : (
              <div style={{ color: 'var(--muted)', fontSize: 13, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
                <Loader2 size={22} className="animate-spin" /> 本地引擎生成中…
              </div>
            )}
          </div>
        </div>
      )}

      {/* 会话画廊 */}
      {gallery.length > 0 && (
        <div style={{ marginTop: 22 }}>
          <div style={{ fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: 0.6, marginBottom: 10 }}>本次会话</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 10 }}>
            {gallery.map((g, i) => (
              <div key={i} title={g.prompt} style={{ aspectRatio: '1', borderRadius: 8, overflow: 'hidden', border: '1px solid var(--border)', background: 'var(--bg-accent)' }}>
                <img src={g.url} alt={g.prompt} style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
      <span style={{ fontSize: 10, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: 0.5 }}>{label}</span>
      {children}
    </div>
  )
}

const selectStyle: React.CSSProperties = {
  background: 'var(--bg)', color: 'var(--text)', border: '1px solid var(--border)',
  borderRadius: 8, padding: '7px 10px', fontSize: 12, maxWidth: 220,
}

function NumBox({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <input
      type="number" step={64} min={256} max={2048} value={value}
      onChange={(e) => onChange(Number(e.target.value) || 1024)}
      style={{ width: 70, background: 'var(--bg)', color: 'var(--text)', border: '1px solid var(--border)', borderRadius: 8, padding: '7px 8px', fontSize: 12 }}
    />
  )
}

/** 客户端搭 Z-Image 文生图工作流图(checkpoint[arch=z-image]→encode→ksampler→vae_decode)。
 *  端口 handle 用 node.yaml 里的端口 id(model/clip/vae/conditioning/latent)。 */
function buildZImageWorkflow(
  { ckpt, prompt, width, height, seed }: { ckpt: string; prompt: string; width: number; height: number; seed: number },
): Workflow {
  const nodes = [
    { id: 'ckpt', type: 'flux2_load_checkpoint' as const, position: { x: 0, y: 0 },
      data: { file: ckpt, weight_dtype: 'bfloat16', device: 'auto', offload: 'none', adapter_arch: 'z-image' } },
    { id: 'enc', type: 'flux2_encode_prompt' as const, position: { x: 320, y: 0 },
      data: { text: prompt, negative_prompt: '' } },
    { id: 'ks', type: 'flux2_ksampler' as const, position: { x: 640, y: 0 },
      data: { width, height, steps: 8, cfg_scale: 1.0, sampler_name: 'euler', scheduler: 'normal', seed: String(seed) } },
    { id: 'dec', type: 'flux2_vae_decode' as const, position: { x: 960, y: 0 }, data: {} },
  ]
  const edges = [
    { id: 'e1', source: 'ckpt', sourceHandle: 'clip', target: 'enc', targetHandle: 'clip' },
    { id: 'e2', source: 'ckpt', sourceHandle: 'model', target: 'ks', targetHandle: 'model' },
    { id: 'e3', source: 'enc', sourceHandle: 'conditioning', target: 'ks', targetHandle: 'conditioning' },
    { id: 'e4', source: 'ckpt', sourceHandle: 'vae', target: 'dec', targetHandle: 'vae' },
    { id: 'e5', source: 'ks', sourceHandle: 'latent', target: 'dec', targetHandle: 'latent' },
  ]
  return { name: '创作台·文生图(Z-Image)', nodes, edges } as unknown as Workflow
}
