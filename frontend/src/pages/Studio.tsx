import { useEffect, useMemo, useRef, useState } from 'react'
import { ImageIcon, Sparkles, Wand2, Scan, Box, Loader2, Monitor, Cloud, Upload } from 'lucide-react'
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
  { id: 'edit', label: '图片编辑', icon: Wand2, ready: true },
  { id: 'enhance', label: '细节增强', icon: Sparkles, ready: true },
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
        {feature === 'text2img' && <Text2ImagePanel />}
        {feature === 'edit' && <ImageEditPanel />}
        {feature === 'enhance' && <EnhancePanel />}
        {feature === 'angle' && (
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
    submitImageWorkflow(wf, {
      onImage: (url) => {
        setResult(url)
        setGallery((g) => [{ url, prompt: prompt.trim(), seed }, ...g].slice(0, 24))
        setRunning(false)
      },
      onError: (msg) => { toast(msg, 'error'); setRunning(false) },
      onTimeout: () => setRunning(false),
      toast,
    })
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
    // image_output 是 executeWorkflow 必需的输出节点(否则抛「工作流缺少输出节点」)。
    { id: 'out', type: 'image_output' as const, position: { x: 1280, y: 0 }, data: {} },
  ]
  const edges = [
    { id: 'e1', source: 'ckpt', sourceHandle: 'clip', target: 'enc', targetHandle: 'clip' },
    { id: 'e2', source: 'ckpt', sourceHandle: 'model', target: 'ks', targetHandle: 'model' },
    { id: 'e3', source: 'enc', sourceHandle: 'conditioning', target: 'ks', targetHandle: 'conditioning' },
    { id: 'e4', source: 'ckpt', sourceHandle: 'vae', target: 'dec', targetHandle: 'vae' },
    { id: 'e5', source: 'ks', sourceHandle: 'latent', target: 'dec', targetHandle: 'latent' },
    { id: 'e6', source: 'dec', sourceHandle: 'image', target: 'out', targetHandle: 'image' },
  ]
  return { name: '创作台·文生图(Z-Image)', nodes, edges } as unknown as Workflow
}

/** 提交工作流 → 监听 WS 'node-progress' 拿 image_url → 回调。两个 panel 共用,避免重复 WS 接线。
 *  ignoreImageFrom:要忽略其 node_complete.image_url 的输入节点 id —— image_input(上传图)节点会
 *  把上传图回显成一个 node_complete(写盘签 URL),不能当成最终出图(否则编辑模式抓到的是原图)。 */
function submitImageWorkflow(
  wf: Workflow,
  h: {
    onImage: (url: string) => void
    onError: (msg: string) => void
    onTimeout: () => void
    toast: (m: string, t?: 'info' | 'error' | 'success') => void
    ignoreImageFrom?: string
  },
): void {
  let settled = false
  const unbind = () => window.removeEventListener('node-progress', onProgress as EventListener)
  const onProgress = (ev: Event) => {
    const d = (ev as CustomEvent).detail
    if (settled) return
    // 跳过输入源节点(image_input)的回显 —— 它不是生成结果。
    if (d?.node_type === 'image_input' || (h.ignoreImageFrom && d?.node_id === h.ignoreImageFrom)) return
    if (d?.type === 'node_complete' && d?.image_url) {
      settled = true; unbind(); h.onImage(String(d.image_url))
    } else if (d?.type === 'node_error') {
      settled = true; unbind(); h.onError(`生成失败:${d.error ?? d.detail ?? '未知错误'}`)
    }
  }
  window.addEventListener('node-progress', onProgress as EventListener)
  executeWorkflow(wf)
    .then(() => h.toast('已入队本地生成…', 'info'))
    .catch((err: unknown) => {
      settled = true; unbind(); h.onError(`提交失败:${(err as Error)?.message ?? err}`)
    })
  // 兜底:120s 没出图 → 解绑防泄漏(编辑大图 + 首次装模型可能久);真结果靠 WS。
  setTimeout(() => { unbind(); if (!settled) h.onTimeout() }, 120_000)
}

/** 读上传文件 → { dataUri, width, height }(snap 宽高到 64 的倍数,clamp 512..2048,Flux2 友好)。 */
function readUpload(file: File): Promise<{ dataUri: string; width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(new Error('读取文件失败'))
    reader.onload = () => {
      const dataUri = String(reader.result)
      const img = new Image()
      img.onerror = () => reject(new Error('无法解析图片'))
      img.onload = () => {
        const snap = (n: number) => Math.max(512, Math.min(2048, Math.round(n / 64) * 64))
        resolve({ dataUri, width: snap(img.naturalWidth || 1024), height: snap(img.naturalHeight || 1024) })
      }
      img.src = dataUri
    }
    reader.readAsDataURL(file)
  })
}

function ImageEditPanel() {
  const toast = useToastStore((s) => s.add)
  const { data: checkpoints } = useComponents('checkpoint')
  const [prompt, setPrompt] = useState('')
  const [ckpt, setCkpt] = useState<string>('')
  const [upload, setUpload] = useState<{ dataUri: string; width: number; height: number } | null>(null)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement | null>(null)

  // 编辑走 Flux2(多参考编辑;Flux2KleinPipeline 接受 image=)。默认挑 Flux2 整模型。
  const fluxCandidates = useMemo(
    () => (checkpoints ?? []).filter((c) => /flux/i.test(c.filename) || /flux/i.test(c.abs_path)),
    [checkpoints],
  )
  useEffect(() => {
    if (ckpt) return
    const pick = fluxCandidates[0] ?? (checkpoints ?? [])[0]
    if (pick) setCkpt(pick.abs_path)
  }, [ckpt, fluxCandidates, checkpoints])

  const pickFile = async (file: File | undefined) => {
    if (!file) return
    try { setUpload(await readUpload(file)) } catch (e) { toast((e as Error).message, 'error') }
  }

  const run = () => {
    if (running) return
    if (!upload) { toast('先上传一张要编辑的图', 'info'); return }
    if (!prompt.trim()) { toast('描述你想怎么改这张图', 'info'); return }
    if (!ckpt) { toast('没找到可用的 Flux2 整模型(在 diffusers/ 放 Flux2-klein-9B)', 'error'); return }
    setRunning(true)
    setResult(null)
    const seed = Math.floor(Math.random() * 1_000_000_000_000)
    const wf = buildFlux2EditWorkflow({
      ckpt, prompt: prompt.trim(), imageDataUri: upload.dataUri,
      width: upload.width, height: upload.height, seed,
    })
    submitImageWorkflow(wf, {
      onImage: (url) => { setResult(url); setRunning(false) },
      onError: (msg) => { toast(msg, 'error'); setRunning(false) },
      onTimeout: () => setRunning(false),
      toast,
      ignoreImageFrom: 'img',  // image_input 节点 id(见 buildFlux2EditWorkflow)
    })
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 24 }}>
      <div style={{ background: 'var(--bg-accent)', border: '1px solid var(--border)', borderRadius: 12, padding: 18 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
          <span style={{ fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: 1 }}>
            图片编辑 · 本地 Flux2(参考编辑)
          </span>
          <span style={{ fontSize: 11, color: 'var(--ok)', display: 'inline-flex', alignItems: 'center', gap: 5 }}>
            系统就绪 <span style={{ width: 6, height: 6, borderRadius: 3, background: 'var(--ok)' }} />
          </span>
        </div>

        <div style={{ display: 'flex', gap: 16, alignItems: 'stretch' }}>
          {/* 上传区 */}
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            style={{
              width: 200, minHeight: 200, flexShrink: 0, borderRadius: 10, cursor: 'pointer',
              border: `1px ${upload ? 'solid' : 'dashed'} var(--border)`, background: 'var(--bg)',
              color: 'var(--muted)', display: 'flex', alignItems: 'center', justifyContent: 'center',
              overflow: 'hidden', padding: 0,
            }}
          >
            {upload ? (
              <img src={upload.dataUri} alt="待编辑" style={{ maxWidth: '100%', maxHeight: 260, display: 'block' }} />
            ) : (
              <span style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, fontSize: 13 }}>
                <Upload size={22} /> 点击上传图片
              </span>
            )}
          </button>
          <input
            ref={fileRef} type="file" accept="image/*" hidden
            onChange={(e) => pickFile(e.target.files?.[0])}
          />

          {/* 控制 */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 12 }}>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="描述编辑指令,如「把背景换成雪景」「让它变成夜晚」…"
              rows={3}
              style={{
                width: '100%', resize: 'vertical', background: 'var(--bg)', color: 'var(--text)',
                border: '1px solid var(--border)', borderRadius: 8, padding: '12px 14px',
                fontSize: 15, lineHeight: 1.5, outline: 'none',
              }}
            />
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 18, flexWrap: 'wrap' }}>
              <Field label="模型(Flux2 整模型)">
                <select value={ckpt} onChange={(e) => setCkpt(e.target.value)} style={selectStyle}>
                  {(checkpoints ?? []).length === 0 && <option value="">无可用整模型</option>}
                  {(checkpoints ?? []).map((c) => (
                    <option key={c.abs_path} value={c.abs_path}>{c.filename}</option>
                  ))}
                </select>
              </Field>
              {upload && (
                <Field label="输出尺寸(跟随原图)">
                  <span style={{ fontSize: 12, color: 'var(--text)' }}>{upload.width} × {upload.height}</span>
                </Field>
              )}
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
                {running ? <Loader2 size={16} className="animate-spin" /> : <Wand2 size={16} />}
                {running ? '本地编辑中…' : '本地编辑'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* 前后对比 */}
      {(running || result) && (
        <div style={{ marginTop: 18, display: 'flex', gap: 16, justifyContent: 'center', flexWrap: 'wrap' }}>
          <ComparePane label="原图" src={upload?.dataUri ?? null} />
          <ComparePane label="编辑后" src={result} loading={running && !result} />
        </div>
      )}
    </div>
  )
}

function EnhancePanel() {
  const toast = useToastStore((s) => s.add)
  const [upload, setUpload] = useState<{ dataUri: string; width: number; height: number } | null>(null)
  const [resolution, setResolution] = useState(1440)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement | null>(null)

  const pickFile = async (file: File | undefined) => {
    if (!file) return
    try { setUpload(await readUpload(file)) } catch (e) { toast((e as Error).message, 'error') }
  }

  const run = () => {
    if (running) return
    if (!upload) { toast('先上传一张要增强的图', 'info'); return }
    setRunning(true)
    setResult(null)
    const wf = buildSeedVR2Workflow({ imageDataUri: upload.dataUri, resolution })
    submitImageWorkflow(wf, {
      onImage: (url) => { setResult(url); setRunning(false) },
      onError: (msg) => { toast(msg, 'error'); setRunning(false) },
      onTimeout: () => setRunning(false),
      toast,
      ignoreImageFrom: 'img',
    })
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 24 }}>
      <div style={{ background: 'var(--bg-accent)', border: '1px solid var(--border)', borderRadius: 12, padding: 18 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
          <span style={{ fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: 1 }}>
            细节增强 · 本地 SeedVR2(超分)
          </span>
          <span style={{ fontSize: 11, color: 'var(--ok)', display: 'inline-flex', alignItems: 'center', gap: 5 }}>
            系统就绪 <span style={{ width: 6, height: 6, borderRadius: 3, background: 'var(--ok)' }} />
          </span>
        </div>
        <div style={{ display: 'flex', gap: 16, alignItems: 'stretch' }}>
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            style={{
              width: 200, minHeight: 160, flexShrink: 0, borderRadius: 10, cursor: 'pointer',
              border: `1px ${upload ? 'solid' : 'dashed'} var(--border)`, background: 'var(--bg)',
              color: 'var(--muted)', display: 'flex', alignItems: 'center', justifyContent: 'center',
              overflow: 'hidden', padding: 0,
            }}
          >
            {upload ? (
              <img src={upload.dataUri} alt="待增强" style={{ maxWidth: '100%', maxHeight: 260, display: 'block' }} />
            ) : (
              <span style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, fontSize: 13 }}>
                <Upload size={22} /> 点击上传图片
              </span>
            )}
          </button>
          <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e) => pickFile(e.target.files?.[0])} />

          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 14, justifyContent: 'center' }}>
            <div style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.6 }}>
              SeedVR2 一步超分:放大并补充细节(无需 prompt)。{upload && ` 原图 ${upload.width}×${upload.height}。`}
            </div>
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 18, flexWrap: 'wrap' }}>
              <Field label={`目标分辨率(短边 ${resolution}px)`}>
                <input
                  type="range" min={512} max={2160} step={16} value={resolution}
                  onChange={(e) => setResolution(Number(e.target.value))}
                  style={{ width: 200 }}
                />
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
                {running ? '本地增强中…' : '本地增强'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {(running || result) && (
        <div style={{ marginTop: 18, display: 'flex', gap: 16, justifyContent: 'center', flexWrap: 'wrap' }}>
          <ComparePane label="原图" src={upload?.dataUri ?? null} />
          <ComparePane label="增强后" src={result} loading={running && !result} />
        </div>
      )}
    </div>
  )
}

function ComparePane({ label, src, loading }: { label: string; src: string | null; loading?: boolean }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <span style={{ fontSize: 10, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: 0.5 }}>{label}</span>
      <div style={{
        width: 360, height: 360, borderRadius: 12, border: '1px solid var(--border)',
        background: 'var(--bg-accent)', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden',
      }}>
        {src ? (
          <img src={src} alt={label} style={{ maxWidth: '100%', maxHeight: '100%', display: 'block' }} />
        ) : loading ? (
          <div style={{ color: 'var(--muted)', fontSize: 13, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
            <Loader2 size={22} className="animate-spin" /> 本地引擎编辑中…
          </div>
        ) : (
          <span style={{ color: 'var(--muted)', fontSize: 12 }}>—</span>
        )}
      </div>
    </div>
  )
}

/** 客户端搭 Flux2 图片编辑工作流图:image_input(上传图)+ checkpoint[arch=flux2] → encode →
 *  ksampler(image 端口接上传图)→ vae_decode。引擎把输入图注入 Flux2KleinPipeline image=(参考编辑)。 */
function buildFlux2EditWorkflow(
  { ckpt, prompt, imageDataUri, width, height, seed }:
  { ckpt: string; prompt: string; imageDataUri: string; width: number; height: number; seed: number },
): Workflow {
  const nodes = [
    { id: 'img', type: 'image_input' as const, position: { x: 0, y: 240 },
      data: { image: imageDataUri } },
    { id: 'ckpt', type: 'flux2_load_checkpoint' as const, position: { x: 0, y: 0 },
      data: { file: ckpt, weight_dtype: 'bfloat16', device: 'auto', offload: 'none', adapter_arch: 'flux2' } },
    { id: 'enc', type: 'flux2_encode_prompt' as const, position: { x: 320, y: 0 },
      data: { text: prompt, negative_prompt: '' } },
    { id: 'ks', type: 'flux2_ksampler' as const, position: { x: 640, y: 0 },
      data: { width, height, steps: 20, cfg_scale: 4.0, sampler_name: 'euler', scheduler: 'normal', seed: String(seed) } },
    { id: 'dec', type: 'flux2_vae_decode' as const, position: { x: 960, y: 0 }, data: {} },
    { id: 'out', type: 'image_output' as const, position: { x: 1280, y: 0 }, data: {} },
  ]
  const edges = [
    { id: 'e1', source: 'ckpt', sourceHandle: 'clip', target: 'enc', targetHandle: 'clip' },
    { id: 'e2', source: 'ckpt', sourceHandle: 'model', target: 'ks', targetHandle: 'model' },
    { id: 'e3', source: 'enc', sourceHandle: 'conditioning', target: 'ks', targetHandle: 'conditioning' },
    { id: 'e4', source: 'img', sourceHandle: 'image', target: 'ks', targetHandle: 'image' },
    { id: 'e5', source: 'ckpt', sourceHandle: 'vae', target: 'dec', targetHandle: 'vae' },
    { id: 'e6', source: 'ks', sourceHandle: 'latent', target: 'dec', targetHandle: 'latent' },
    { id: 'e7', source: 'dec', sourceHandle: 'image', target: 'out', targetHandle: 'image' },
  ]
  return { name: '创作台·图片编辑(Flux2)', nodes, edges } as unknown as Workflow
}

/** 客户端搭 SeedVR2 细节增强(超分)工作流图:image_input(上传图)→ seedvr2_upscale → image_output。
 *  dit/vae loader 不连 → runner 用默认 DiT/VAE(见 seedvr2/node.yaml「不连则用默认」)。
 *  seedvr2_upscale 是 dispatch 节点(role=upscale),runner 构 UpscaleRequest 跑 SeedVR2 一步超分。 */
function buildSeedVR2Workflow(
  { imageDataUri, resolution }: { imageDataUri: string; resolution: number },
): Workflow {
  const nodes = [
    { id: 'img', type: 'image_input' as const, position: { x: 0, y: 0 }, data: { image: imageDataUri } },
    { id: 'up', type: 'seedvr2_upscale' as const, position: { x: 320, y: 0 },
      data: { resolution, max_resolution: 0, color_correction: 'lab' } },
    { id: 'out', type: 'image_output' as const, position: { x: 640, y: 0 }, data: {} },
  ]
  const edges = [
    { id: 'e1', source: 'img', sourceHandle: 'image', target: 'up', targetHandle: 'image' },
    { id: 'e2', source: 'up', sourceHandle: 'image', target: 'out', targetHandle: 'image' },
  ]
  return { name: '创作台·细节增强(SeedVR2)', nodes, edges } as unknown as Workflow
}
