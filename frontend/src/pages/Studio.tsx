import { useEffect, useMemo, useRef, useState } from 'react'
import { ImageIcon, Sparkles, Wand2, Box, Loader2, Monitor, Cloud, Upload, Share2, Layers, Plus, Trash2 } from 'lucide-react'
import { useComponents } from '../api/components'
import { executeWorkflow } from '../utils/workflowExecutor'
import type { Workflow } from '../models/workflow'
import { useToastStore } from '../stores/toast'
import { useCreateWorkflow } from '../api/workflows'
import { usePublishWorkflow } from '../api/services'
import {
  buildZImageWorkflow, buildFlux2EditWorkflow, buildSeedVR2Workflow, buildQwenEditWorkflow,
  buildFeatureWorkflow, buildChainWorkflow, FEATURE_PUBLISH,
  type FeatureId as PublishFeatureId, type ChainStage,
} from './studioWorkflows'

// 创作台:对齐 Infinite-Canvas 的「统一创作控制台」,但引擎是 nous 本地(不外接 ComfyUI)。
// 客户端搭 z-image 工作流图 → /api/v1/workflows/execute(admin cookie 鉴权)→ WS 拿 image_url。
// 四功能:文生图(Z-Image,已通)/ 细节增强 / 图片编辑 / 角度控制(后三个随 P2/P3 引擎上线接)。

type FeatureId = 'text2img' | 'enhance' | 'edit' | 'angle' | 'chain'

const FEATURES: { id: FeatureId; label: string; icon: typeof ImageIcon; ready: boolean }[] = [
  { id: 'text2img', label: '文生图', icon: ImageIcon, ready: true },
  { id: 'edit', label: '图片编辑', icon: Wand2, ready: true },
  { id: 'enhance', label: '细节增强', icon: Sparkles, ready: true },
  { id: 'angle', label: '角度控制', icon: Box, ready: true },
  { id: 'chain', label: '链式采样', icon: Layers, ready: true },
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
        {feature === 'angle' && <AnglePanel />}
        {feature === 'chain' && <ChainSamplePanel />}
      </div>
    </div>
  )
}

// 按功能自动挑整模型(同各 panel 的选模型正则);enhance(SeedVR2)用默认 DiT/VAE 无需 checkpoint。
const CKPT_MATCH: Record<PublishFeatureId, RegExp | null> = {
  text2img: /z[-_ ]?image/i,
  edit: /flux/i,
  angle: /qwen[-_ ]?image[-_ ]?edit/i,
  enhance: null,
}

/** 「发布为服务」按钮 + 弹窗:存工作流(模板)→ publish(带 exposed schema)→ 外部 /v1/images/generations 可调。
 *  自包含:按功能自动挑 checkpoint(无需 panel 传参),schema 来自 studioWorkflows.FEATURE_PUBLISH。 */
function PublishServiceButton({ feature }: { feature: PublishFeatureId }) {
  const toast = useToastStore((s) => s.add)
  const { data: checkpoints } = useComponents('checkpoint')
  const createWf = useCreateWorkflow()
  const publish = usePublishWorkflow()
  const conf = FEATURE_PUBLISH[feature]
  const [open, setOpen] = useState(false)
  const [name, setName] = useState(conf.defaultName)
  const busy = createWf.isPending || publish.isPending

  const re = CKPT_MATCH[feature]
  const ckpt = useMemo(() => {
    if (re == null) return '' // enhance 不需要整模型
    const hit = (checkpoints ?? []).find((c) => re.test(c.filename) || re.test(c.abs_path))
    return hit?.abs_path ?? ''
  }, [checkpoints, re])

  const doPublish = async () => {
    if (!/^[a-z][a-z0-9-]{1,62}$/.test(name)) { toast('服务名须匹配 ^[a-z][a-z0-9-]{1,62}$', 'error'); return }
    if (re != null && !ckpt) { toast('没找到该功能可用的整模型(检查 diffusers/ 目录)', 'error'); return }
    try {
      const wf = buildFeatureWorkflow(feature, ckpt) as unknown as { nodes: unknown[]; edges: unknown[] }
      const created = await createWf.mutateAsync({ name: conf.label, nodes: wf.nodes as never[], edges: wf.edges as never[] })
      await publish.mutateAsync({
        workflowId: created.id,
        body: {
          name, label: conf.label, category: conf.category,
          exposed_inputs: conf.exposed_inputs, exposed_outputs: conf.exposed_outputs,
        },
      })
      toast(`已发布服务「${name}」—— 外部 POST /v1/images/generations(model=${name})可调用`, 'success')
      setOpen(false)
    } catch (e) {
      toast(`发布失败:${(e as Error)?.message ?? e}`, 'error')
    }
  }

  return (
    <>
      <button
        type="button"
        onClick={() => { setName(conf.defaultName); setOpen(true) }}
        style={{
          display: 'inline-flex', alignItems: 'center', gap: 6, padding: '7px 12px',
          borderRadius: 8, fontSize: 12.5, border: '1px solid var(--border)',
          background: 'var(--bg-accent)', color: 'var(--text)', cursor: 'pointer',
        }}
      >
        <Share2 size={14} /> 发布为服务
      </button>
      {open && (
        <div
          onClick={() => !busy && setOpen(false)}
          style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)', zIndex: 50, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{ width: 420, background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 12, padding: 20 }}
          >
            <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 4 }}>发布「{conf.label}」为服务</div>
            <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 14 }}>
              冻结当前工作流为外部 API 服务,经 POST /v1/images/generations 调用(model=服务名)。
              {re != null && !ckpt && <span style={{ color: 'var(--danger, #e5484d)' }}> 未找到整模型!</span>}
            </div>
            <label style={{ fontSize: 12, color: 'var(--muted)' }}>服务名(小写字母/数字/连字符)</label>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={busy}
              style={{ width: '100%', marginTop: 6, marginBottom: 8, padding: '8px 10px', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-accent)', color: 'var(--text)', fontSize: 13 }}
            />
            <div style={{ fontSize: 11.5, color: 'var(--muted)', marginBottom: 16 }}>
              暴露入参:{conf.exposed_inputs.map((p) => p.key).join(' / ')}
              {conf.category === 'image' && '(category=image)'}
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
              <button type="button" onClick={() => !busy && setOpen(false)} disabled={busy}
                style={{ padding: '8px 14px', borderRadius: 8, border: '1px solid var(--border)', background: 'transparent', color: 'var(--text)', cursor: busy ? 'default' : 'pointer', fontSize: 13 }}>
                取消
              </button>
              <button type="button" onClick={doPublish} disabled={busy}
                style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '8px 14px', borderRadius: 8, border: 'none', background: 'var(--text)', color: 'var(--bg)', cursor: busy ? 'default' : 'pointer', fontSize: 13, fontWeight: 600 }}>
                {busy && <Loader2 size={14} className="animate-spin" />} 发布
              </button>
            </div>
          </div>
        </div>
      )}
    </>
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
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 12 }}>
            <PublishServiceButton feature="text2img" />
            <span style={{ fontSize: 11, color: 'var(--ok)', display: 'inline-flex', alignItems: 'center', gap: 5 }}>
              系统就绪 <span style={{ width: 6, height: 6, borderRadius: 3, background: 'var(--ok)' }} />
            </span>
          </div>
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

/** 链式采样提交:每段终端(flux2_vae_decode)各 emit 一个 node_complete.image_url —— 按终端 id
 *  映射回段序号,逐段回调 onStage(i,url);最终段完成才 onDone。多终端,不能用 submitImageWorkflow
 *  (它抓第一个 image_url 就 settle = 只拿到第一段)。 */
function submitChainWorkflow(
  wf: Workflow,
  terminals: string[],
  h: {
    onStage: (i: number, url: string) => void
    onDone: () => void
    onError: (msg: string) => void
    onTimeout: () => void
    toast: (m: string, t?: 'info' | 'error' | 'success') => void
  },
): void {
  let settled = false
  const lastId = terminals[terminals.length - 1]
  const idx = new Map(terminals.map((t, i) => [t, i]))
  const unbind = () => window.removeEventListener('node-progress', onProgress as EventListener)
  const onProgress = (ev: Event) => {
    const d = (ev as CustomEvent).detail
    if (settled) return
    if (d?.type === 'node_complete' && d?.image_url && idx.has(d.node_id)) {
      h.onStage(idx.get(d.node_id)!, String(d.image_url))
      if (d.node_id === lastId) { settled = true; unbind(); h.onDone() }
    } else if (d?.type === 'node_error') {
      settled = true; unbind(); h.onError(`生成失败:${d.error ?? d.detail ?? '未知错误'}`)
    }
  }
  window.addEventListener('node-progress', onProgress as EventListener)
  executeWorkflow(wf)
    .then(() => h.toast('已入队链式采样…', 'info'))
    .catch((err: unknown) => { settled = true; unbind(); h.onError(`提交失败:${(err as Error)?.message ?? err}`) })
  // 链多段 + 可能多模型首装,放宽到 300s。
  setTimeout(() => { unbind(); if (!settled) h.onTimeout() }, 300_000)
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
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 12 }}>
            <PublishServiceButton feature="edit" />
            <span style={{ fontSize: 11, color: 'var(--ok)', display: 'inline-flex', alignItems: 'center', gap: 5 }}>
              系统就绪 <span style={{ width: 6, height: 6, borderRadius: 3, background: 'var(--ok)' }} />
            </span>
          </div>
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
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 12 }}>
            <PublishServiceButton feature="enhance" />
            <span style={{ fontSize: 11, color: 'var(--ok)', display: 'inline-flex', alignItems: 'center', gap: 5 }}>
              系统就绪 <span style={{ width: 6, height: 6, borderRadius: 3, background: 'var(--ok)' }} />
            </span>
          </div>
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

const ANGLE_PRESETS: { label: string; prompt: string }[] = [
  { label: '左前 45°', prompt: 'rotate the camera to show the subject from a 45-degree front-left angle, keep the same subject and style' },
  { label: '右前 45°', prompt: 'rotate the camera to show the subject from a 45-degree front-right angle, keep the same subject and style' },
  { label: '左侧面', prompt: 'show the subject from the left side profile view, keep the same subject and style' },
  { label: '右侧面', prompt: 'show the subject from the right side profile view, keep the same subject and style' },
  { label: '背面', prompt: 'show the subject from the back / rear view, keep the same subject and style' },
  { label: '俯视', prompt: 'show the subject from a top-down bird\'s-eye view, keep the same subject and style' },
  { label: '仰视', prompt: 'show the subject from a low worm\'s-eye angle looking up, keep the same subject and style' },
]

function AnglePanel() {
  const toast = useToastStore((s) => s.add)
  const { data: checkpoints } = useComponents('checkpoint')
  const [ckpt, setCkpt] = useState<string>('')
  const [upload, setUpload] = useState<{ dataUri: string; width: number; height: number } | null>(null)
  const [angleIdx, setAngleIdx] = useState(0)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement | null>(null)

  // 角度控制走 Qwen-Image-Edit-2511(编辑类,needs_image_input)。自动挑 Qwen-Edit 整模型。
  const qwenCandidates = useMemo(
    () => (checkpoints ?? []).filter((c) => /qwen.*image.*edit|qwen.*edit/i.test(c.filename) || /qwen.*image.*edit/i.test(c.abs_path)),
    [checkpoints],
  )
  useEffect(() => {
    if (ckpt) return
    const pick = qwenCandidates[0] ?? (checkpoints ?? [])[0]
    if (pick) setCkpt(pick.abs_path)
  }, [ckpt, qwenCandidates, checkpoints])

  const pickFile = async (file: File | undefined) => {
    if (!file) return
    try { setUpload(await readUpload(file)) } catch (e) { toast((e as Error).message, 'error') }
  }

  const run = () => {
    if (running) return
    if (!upload) { toast('先上传一张图', 'info'); return }
    if (!ckpt) { toast('没找到 Qwen-Image-Edit 整模型(在 diffusers/ 放 Qwen-Image-Edit-2511)', 'error'); return }
    setRunning(true)
    setResult(null)
    const seed = Math.floor(Math.random() * 1_000_000_000_000)
    const wf = buildQwenEditWorkflow({
      ckpt, prompt: ANGLE_PRESETS[angleIdx].prompt, imageDataUri: upload.dataUri,
      width: upload.width, height: upload.height, seed,
    })
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
            角度控制 · 本地 Qwen-Image-Edit
          </span>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 12 }}>
            <PublishServiceButton feature="angle" />
            <span style={{ fontSize: 11, color: 'var(--ok)', display: 'inline-flex', alignItems: 'center', gap: 5 }}>
              系统就绪 <span style={{ width: 6, height: 6, borderRadius: 3, background: 'var(--ok)' }} />
            </span>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 16, alignItems: 'stretch' }}>
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
              <img src={upload.dataUri} alt="原图" style={{ maxWidth: '100%', maxHeight: 260, display: 'block' }} />
            ) : (
              <span style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, fontSize: 13 }}>
                <Upload size={22} /> 点击上传图片
              </span>
            )}
          </button>
          <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e) => pickFile(e.target.files?.[0])} />

          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 12 }}>
            <Field label="目标视角">
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {ANGLE_PRESETS.map((a, i) => (
                  <button
                    key={a.label}
                    type="button"
                    onClick={() => setAngleIdx(i)}
                    style={{
                      padding: '7px 12px', borderRadius: 8, fontSize: 12, cursor: 'pointer',
                      border: '1px solid var(--border)',
                      background: i === angleIdx ? 'var(--text)' : 'var(--bg)',
                      color: i === angleIdx ? 'var(--bg)' : 'var(--text)',
                      fontWeight: i === angleIdx ? 600 : 400,
                    }}
                  >
                    {a.label}
                  </button>
                ))}
              </div>
            </Field>
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 18, flexWrap: 'wrap' }}>
              <Field label="模型(Qwen-Edit 整模型)">
                <select value={ckpt} onChange={(e) => setCkpt(e.target.value)} style={selectStyle}>
                  {(checkpoints ?? []).length === 0 && <option value="">无可用整模型</option>}
                  {(checkpoints ?? []).map((c) => (
                    <option key={c.abs_path} value={c.abs_path}>{c.filename}</option>
                  ))}
                </select>
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
                {running ? <Loader2 size={16} className="animate-spin" /> : <Box size={16} />}
                {running ? '本地生成中…' : '本地生成'}
              </button>
            </div>
            <div style={{ fontSize: 11, color: 'var(--muted)' }}>
              Qwen-Image-Edit 改相机视角,保持主体不变。20B 模型,首次加载较久。
            </div>
          </div>
        </div>
      </div>

      {(running || result) && (
        <div style={{ marginTop: 18, display: 'flex', gap: 16, justifyContent: 'center', flexWrap: 'wrap' }}>
          <ComparePane label="原图" src={upload?.dataUri ?? null} />
          <ComparePane label={ANGLE_PRESETS[angleIdx].label} src={result} loading={running && !result} />
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

// 链式采样架构选项 + 自动挑 checkpoint 的正则(对齐各 panel 的选模型正则)。
const ARCH_OPTIONS: { arch: string; label: string; re: RegExp; canRef: boolean }[] = [
  { arch: 'z-image', label: 'Z-Image(文生图)', re: /z[-_ ]?image/i, canRef: false },
  { arch: 'flux2', label: 'Flux2-Klein(参考编辑)', re: /flux2|flux[-_ ]?klein/i, canRef: true },
  { arch: 'qwen-edit', label: 'Qwen-Edit(参考编辑)', re: /qwen.*edit/i, canRef: true },
]

interface ChainStageUI extends ChainStage { result: string | null }

/** 链式采样(跨模型 A-ref):多段采样,每段把上一段出图作参考编辑条件喂下一段。
 *  真机已验 Z-Image→Flux2-Klein→Flux2-Klein 三采样(零引擎改;spec 2026-06-08-multi-sampling-cross-model)。 */
function ChainSamplePanel() {
  const toast = useToastStore((s) => s.add)
  const { data: checkpoints } = useComponents('checkpoint')
  const [width, setWidth] = useState(1024)
  const [height, setHeight] = useState(1024)
  const [running, setRunning] = useState(false)
  const [stages, setStages] = useState<ChainStageUI[]>([
    { ckpt: '', arch: 'z-image', prompt: '', steps: 8, cfg: 1, result: null },
    { ckpt: '', arch: 'flux2', prompt: '', steps: 20, cfg: 4, result: null },
  ])

  // 按 arch 自动挑一个 checkpoint(abs_path),用户可改。
  const pickCkpt = (arch: string): string => {
    const opt = ARCH_OPTIONS.find((o) => o.arch === arch)
    const list = checkpoints ?? []
    const hit = opt ? list.find((c) => opt.re.test(c.filename) || opt.re.test(c.abs_path)) : undefined
    return (hit ?? list[0])?.abs_path ?? ''
  }
  // checkpoints 加载后,给未选 ckpt 的段填默认。
  useEffect(() => {
    if (!checkpoints?.length) return
    setStages((prev) => prev.map((s) => (s.ckpt ? s : { ...s, ckpt: pickCkpt(s.arch) })))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [checkpoints])

  const setStage = (i: number, patch: Partial<ChainStageUI>) =>
    setStages((prev) => prev.map((s, j) => (j === i ? { ...s, ...patch } : s)))
  const addStage = () => setStages((prev) => [
    ...prev,
    { ckpt: pickCkpt('flux2'), arch: 'flux2', prompt: '', steps: 20, cfg: 4, result: null },
  ])
  const removeStage = (i: number) => setStages((prev) => prev.filter((_, j) => j !== i))

  const run = async () => {
    if (running) return
    if (stages.length < 2) { toast('链式采样至少 2 段', 'info'); return }
    if (stages.some((s) => !s.prompt.trim())) { toast('每段都要填提示词', 'info'); return }
    if (stages.some((s) => !s.ckpt)) { toast('有段没选到整模型(在 diffusers/ 放对应模型)', 'error'); return }
    setRunning(true)
    setStages((prev) => prev.map((s) => ({ ...s, result: null })))
    const seed = Math.floor(Math.random() * 1_000_000_000)
    const chainStages: ChainStage[] = stages.map((s) => ({
      ckpt: s.ckpt, arch: s.arch, prompt: s.prompt.trim(), steps: s.steps, cfg: s.cfg }))
    const { workflow, stageTerminals } = buildChainWorkflow(chainStages, { width, height, seed })
    submitChainWorkflow(workflow, stageTerminals, {
      onStage: (i, url) => setStage(i, { result: url }),
      onDone: () => { setRunning(false); toast('链式采样完成', 'success') },
      onError: (msg) => { toast(msg, 'error'); setRunning(false) },
      onTimeout: () => setRunning(false),
      toast,
    })
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 24 }}>
      <div style={{ background: 'var(--bg-accent)', border: '1px solid var(--border)', borderRadius: 12, padding: 18 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
          <span style={{ fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: 1 }}>
            跨模型链式采样 · 本地引擎
          </span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <Field label="尺寸">
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <NumBox value={width} onChange={setWidth} />
                <span style={{ color: 'var(--muted)' }}>×</span>
                <NumBox value={height} onChange={setHeight} />
              </div>
            </Field>
          </div>
        </div>
        <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 14, lineHeight: 1.5 }}>
          第 1 段文生图,后续每段把上一段的出图作「参考编辑」喂下一段(Flux2-Klein / Qwen-Edit 生效;
          Z-Image 无参考编辑能力会忽略上一段图)。
        </div>

        {stages.map((s, i) => {
          const opt = ARCH_OPTIONS.find((o) => o.arch === s.arch)
          const refHint = i > 0 && opt && !opt.canRef
          return (
            <div key={i} style={{
              border: '1px solid var(--border)', borderRadius: 10, padding: 14, marginBottom: 10, background: 'var(--bg)',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                <span style={{
                  width: 22, height: 22, borderRadius: 11, background: 'var(--text)', color: 'var(--bg)',
                  display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, fontWeight: 700,
                }}>{i + 1}</span>
                <span style={{ fontSize: 13, fontWeight: 600 }}>
                  {i === 0 ? '起始(文生图)' : `第 ${i + 1} 段(参考编辑上一段)`}
                </span>
                <div style={{ flex: 1 }} />
                {stages.length > 2 && (
                  <button type="button" onClick={() => removeStage(i)} title="删除该段"
                    style={{ background: 'transparent', border: 'none', color: 'var(--muted)', cursor: 'pointer', padding: 4 }}>
                    <Trash2 size={15} />
                  </button>
                )}
              </div>
              <textarea
                value={s.prompt}
                onChange={(e) => setStage(i, { prompt: e.target.value })}
                placeholder={i === 0 ? '描述起始画面…' : '描述本段如何改造上一段出图…'}
                rows={2}
                style={{
                  width: '100%', resize: 'vertical', background: 'var(--bg-accent)', color: 'var(--text)',
                  border: '1px solid var(--border)', borderRadius: 8, padding: '9px 12px', fontSize: 14, outline: 'none',
                }}
              />
              <div style={{ display: 'flex', alignItems: 'flex-end', gap: 14, marginTop: 10, flexWrap: 'wrap' }}>
                <Field label="架构">
                  <select value={s.arch} onChange={(e) => setStage(i, { arch: e.target.value, ckpt: pickCkpt(e.target.value) })} style={selectStyle}>
                    {ARCH_OPTIONS.map((o) => <option key={o.arch} value={o.arch}>{o.label}</option>)}
                  </select>
                </Field>
                <Field label="模型(整模型)">
                  <select value={s.ckpt} onChange={(e) => setStage(i, { ckpt: e.target.value })} style={selectStyle}>
                    {(checkpoints ?? []).length === 0 && <option value="">无可用整模型</option>}
                    {(checkpoints ?? []).map((c) => (
                      <option key={c.abs_path} value={c.abs_path}>{c.filename}</option>
                    ))}
                  </select>
                </Field>
                <Field label="步数">
                  <NumBox value={s.steps} onChange={(v) => setStage(i, { steps: v })} />
                </Field>
                <Field label="CFG">
                  <NumBox value={s.cfg} onChange={(v) => setStage(i, { cfg: v })} />
                </Field>
              </div>
              {refHint && (
                <div style={{ fontSize: 11, color: 'var(--danger, #e5484d)', marginTop: 8 }}>
                  Z-Image 不接受参考图 —— 这段会忽略上一段出图,只按提示词文生图。要接力请选 Flux2-Klein / Qwen-Edit。
                </div>
              )}
            </div>
          )
        })}

        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 4 }}>
          <button type="button" onClick={addStage}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 6, padding: '8px 14px',
              background: 'var(--bg)', color: 'var(--text)', border: '1px dashed var(--border)',
              borderRadius: 8, fontSize: 13, cursor: 'pointer',
            }}>
            <Plus size={15} /> 添加一段
          </button>
          <div style={{ flex: 1 }} />
          <button type="button" onClick={run} disabled={running}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 8, padding: '11px 22px',
              background: 'var(--text)', color: 'var(--bg)', border: 'none', borderRadius: 8,
              fontSize: 14, fontWeight: 600, cursor: running ? 'wait' : 'pointer', opacity: running ? 0.7 : 1,
            }}>
            {running ? <Loader2 size={16} className="animate-spin" /> : <Layers size={16} />}
            {running ? '链式采样中…' : `运行 ${stages.length} 段链`}
          </button>
        </div>
      </div>

      {/* 逐段结果(横向接力展示) */}
      {(running || stages.some((s) => s.result)) && (
        <div style={{ marginTop: 18, display: 'flex', gap: 12, justifyContent: 'center', flexWrap: 'wrap' }}>
          {stages.map((s, i) => (
            <ComparePane key={i} label={`第 ${i + 1} 段`} src={s.result} loading={running && !s.result} />
          ))}
        </div>
      )}
    </div>
  )
}

// 四功能的 build 函数 + 发布 schema 已抽到 ./studioWorkflows(纯模块,可测,防 schema 漂移)。
