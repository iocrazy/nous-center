import { useState, useCallback, useEffect } from 'react'
import { Copy, Check } from 'lucide-react'
import {
  useEngines, useLoadEngine, useUnloadEngine, useSyncMetadata,
  useScanModels, useSetResident, useRefreshMetadata, useGpus, useSetGpu,
  type EngineInfo,
} from '../../api/engines'
import { apiFetch } from '../../api/client'
import { useToastStore } from '../../stores/toast'
import ContextMenu, { type MenuItem } from '../ui/ContextMenu'

const TYPE_LABELS: Record<string, string> = {
  llm: '语言模型 LLM',
  tts: '语音合成 TTS',
  image: '图像生成 Image',
  video: '视频生成 Video',
  understand: '多模态理解 VL',
}

const TYPE_ORDER = ['llm', 'tts', 'image', 'video', 'understand']

// m11-style tag colors per model type — keeps semantic differentiation
// without the eye-watering rainbow of the legacy "everything is a chip" UI.
const TYPE_TAG_STYLE: Record<string, { bg: string; color: string }> = {
  llm:        { bg: 'var(--accent-2-subtle)', color: 'var(--accent-2)' },                           // green/teal
  tts:        { bg: 'rgba(168,85,247,0.15)',  color: 'rgb(196, 154, 247)' },                        // purple
  image:      { bg: 'rgba(20,184,166,0.18)',  color: 'var(--accent-2)' },                           // teal (slightly stronger)
  video:      { bg: 'rgba(244,114,182,0.15)', color: 'rgb(244,114,182)' },                          // pink
  understand: { bg: 'rgba(59,130,246,0.15)',  color: 'var(--info, #3b82f6)' },                      // blue
}

// Short labels for tabs (vs full names in section headers)
const TAB_LABELS: Record<string, string> = {
  llm: '语言模型',
  tts: '语音合成',
  image: '图像',
  video: '视频',
  understand: '视觉',
}

type TabId = 'all' | 'loaded' | typeof TYPE_ORDER[number]

interface ContextMenuState {
  visible: boolean
  position: { x: number; y: number }
  model: EngineInfo | null
}

export default function ModelsOverlay() {
  const { data: engines, isLoading, isError } = useEngines()
  const loadEngine = useLoadEngine()
  const unloadEngine = useUnloadEngine()
  const syncMeta = useSyncMetadata()
  const scanModels = useScanModels()
  const setResident = useSetResident()
  const refreshMeta = useRefreshMetadata()
  const { data: gpuData } = useGpus()
  const setGpu = useSetGpu()

  const [ctxMenu, setCtxMenu] = useState<ContextMenuState>({
    visible: false,
    position: { x: 0, y: 0 },
    model: null,
  })
  const [activeTab, setActiveTab] = useState<TabId>('all')

  const closeMenu = useCallback(() => {
    setCtxMenu((prev) => ({ ...prev, visible: false }))
  }, [])

  const handleContextMenu = useCallback((e: React.MouseEvent, model: EngineInfo) => {
    e.preventDefault()
    setCtxMenu({ visible: true, position: { x: e.clientX, y: e.clientY }, model })
  }, [])

  const handleToggle = useCallback(
    (engine: EngineInfo) => {
      if (engine.status === 'loading') return // ignore while loading
      if (engine.status === 'loaded') {
        unloadEngine.mutate(engine.name)
        return
      }
      if (!engine.has_adapter) {
        // Auto-detected diffusers without an adapter — backend would 422
        // anyway. Surface the same hint without making the request.
        useToastStore.getState().add(
          `${engine.name} 未注册：图像/视频 adapter 未实现，需要先在 backend/configs/models.yaml 添加 adapter`,
          'error',
        )
        return
      }
      loadEngine.mutate(engine.name)
    },
    [loadEngine, unloadEngine],
  )

  const hasAnyMissing = (engines ?? []).some((e) => !e.has_metadata)

  // Build context menu items for the active model
  const menuItems: MenuItem[] = ctxMenu.model
    ? [
        {
          label: ctxMenu.model.status === 'loaded' ? '卸载模型'
            : ctxMenu.model.status === 'loading' ? '加载中...'
            : !ctxMenu.model.has_adapter ? '未注册（无 adapter）'
            : '加载模型',
          onClick: () => handleToggle(ctxMenu.model!),
          disabled:
            ctxMenu.model.status === 'loading'
            || (ctxMenu.model.status !== 'loaded' && !ctxMenu.model.has_adapter),
        },
        {
          label: ctxMenu.model.resident ? '取消自动加载' : '设为自动加载',
          onClick: () =>
            setResident.mutate({
              name: ctxMenu.model!.name,
              resident: !ctxMenu.model!.resident,
            }),
        },
        { label: '', divider: true },
        {
          label: 'GPU 分配',
          submenu: (gpuData?.devices ?? []).map((g) => {
            const currentGpu = ctxMenu.model!.gpu
            const isCurrentGpu = Array.isArray(currentGpu)
              ? currentGpu.includes(g.index)
              : currentGpu === g.index
            return {
              label: `GPU ${g.index}: ${g.name}`,
              onClick: () => setGpu.mutate({ name: ctxMenu.model!.name, gpu: g.index }),
              disabled: isCurrentGpu,
            }
          }),
        },
        { label: '', divider: true },
        {
          label: '创建 API 接入点',
          onClick: async () => {
            const model = ctxMenu.model!
            try {
              const instance = await apiFetch<{ id: string }>('/api/v1/instances', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  source_type: 'model',
                  source_name: model.name,
                  name: `${model.display_name} API`,
                  type: model.type,
                }),
              })
              // Auto-create an API key
              const keyResult = await apiFetch<{ key: string; id: string }>(`/api/v1/instances/${instance.id}/keys`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ label: 'default' }),
              })
              // Show the key
              window.prompt('API Endpoint Info (Ctrl+C to copy):', keyResult.key)
              useToastStore.getState().add(`接入点已创建: ${model.display_name}`, 'success')
            } catch (e: any) {
              useToastStore.getState().add(`创建失败: ${e.message}`, 'error')
            }
          },
          disabled: ctxMenu.model.status !== 'loaded',
        },
        { label: '', divider: true },
        {
          label: '刷新元数据',
          onClick: () => refreshMeta.mutate(ctxMenu.model!.name),
        },
        {
          label: '删除',
          danger: true,
          disabled: true,
        },
      ]
    : []

  // Compute tab counts (m11 style — single flat grid filtered by tab,
  // not the per-type sectioned rendering of the previous IA).
  const allEngines = engines ?? []
  const loadedCount = allEngines.filter((e) => e.status === 'loaded').length
  const typeCounts: Record<string, number> = {}
  for (const e of allEngines) typeCounts[e.type] = (typeCounts[e.type] ?? 0) + 1

  const visibleEngines = (() => {
    if (activeTab === 'all') return allEngines
    if (activeTab === 'loaded') return allEngines.filter((e) => e.status === 'loaded')
    return allEngines.filter((e) => e.type === activeTab)
  })()

  // Tab list — only show type tabs that have at least one engine
  const tabs: { id: TabId; label: string; count: number }[] = [
    { id: 'all', label: '全部', count: allEngines.length },
    ...TYPE_ORDER
      .filter((t) => (typeCounts[t] ?? 0) > 0)
      .map((t) => ({ id: t as TabId, label: TAB_LABELS[t] ?? t, count: typeCounts[t] })),
    { id: 'loaded', label: '已加载', count: loadedCount },
  ]

  return (
    <div
      className="absolute inset-0 overflow-y-auto z-[16]"
      style={{ background: 'var(--bg)' }}
    >
      <div style={{ padding: '24px 28px', maxWidth: 1600, margin: '0 auto' }}>
        {/* m11 header: title + subtitle on left, toolbar on right */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            marginBottom: 18,
          }}
        >
          <div>
            <h1
              style={{
                fontSize: 20,
                fontWeight: 600,
                color: 'var(--text-strong)',
                margin: 0,
              }}
            >
              引擎库
            </h1>
            <p
              style={{
                fontSize: 13,
                color: 'var(--muted)',
                marginTop: 4,
                marginBottom: 0,
              }}
            >
              底层推理引擎文件与 GPU 常驻管理 · 右键卡片查看更多操作
            </p>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              onClick={() => scanModels.mutate()}
              disabled={scanModels.isPending}
              style={{
                padding: '6px 14px',
                fontSize: 12,
                borderRadius: 4,
                border: '1px solid var(--border)',
                background: 'var(--bg)',
                color: 'var(--text)',
                cursor: scanModels.isPending ? 'wait' : 'pointer',
                opacity: scanModels.isPending ? 0.6 : 1,
              }}
            >
              {scanModels.isPending ? '扫描中...' : '扫描模型'}
            </button>
            {hasAnyMissing && (
              <button
                onClick={() => syncMeta.mutate()}
                disabled={syncMeta.isPending}
                style={{
                  padding: '6px 14px',
                  fontSize: 12,
                  borderRadius: 4,
                  border: 'none',
                  background: 'var(--accent)',
                  color: '#fff',
                  cursor: syncMeta.isPending ? 'wait' : 'pointer',
                  opacity: syncMeta.isPending ? 0.6 : 1,
                }}
              >
                {syncMeta.isPending ? '同步中...' : '+ 拉取模型信息'}
              </button>
            )}
          </div>
        </div>

        {/* m11 tabs row: all / per-type (with non-zero count) / loaded */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            borderBottom: '1px solid var(--border)',
            marginBottom: 18,
            gap: 2,
          }}
        >
          {tabs.map((t) => {
            const isActive = activeTab === t.id
            return (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                style={{
                  padding: '8px 18px',
                  fontSize: 13,
                  fontWeight: isActive ? 600 : 500,
                  background: 'transparent',
                  border: 'none',
                  borderBottom: '2px solid',
                  borderBottomColor: isActive ? 'var(--accent)' : 'transparent',
                  color: isActive ? 'var(--text)' : 'var(--muted)',
                  cursor: 'pointer',
                  marginBottom: -1,
                }}
              >
                {t.label} {t.count}
              </button>
            )
          })}
        </div>

        {isLoading && (
          <div style={{ fontSize: 11, color: 'var(--muted)' }}>加载中...</div>
        )}

        {isError && !engines && (
          <div style={{ fontSize: 11, color: 'var(--muted)' }}>
            无法连接后端服务，等待重试...
          </div>
        )}

        {/* m11 single flat grid (no per-type sections) — tab is the filter */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
            gap: 12,
          }}
        >
          {visibleEngines.map((model) => (
            <ModelCard
              key={model.name}
              model={model}
              onContextMenu={(e) => handleContextMenu(e, model)}
              onToggleResident={(name, resident) => setResident.mutate({ name, resident })}
            />
          ))}
        </div>

        {visibleEngines.length === 0 && !isLoading && (
          <div style={{ fontSize: 11, color: 'var(--muted)', padding: 24, textAlign: 'center' }}>
            该类目没有模型 · 试试别的 tab
          </div>
        )}
      </div>

      {ctxMenu.visible && ctxMenu.model && (
        <ContextMenu items={menuItems} position={ctxMenu.position} onClose={closeMenu} />
      )}
    </div>
  )
}

function ModelCard({
  model,
  onContextMenu,
  onToggleResident,
}: {
  model: EngineInfo
  onContextMenu: (e: React.MouseEvent) => void
  onToggleResident: (name: string, resident: boolean) => void
}) {
  const notDownloaded = model.local_path != null && !model.local_exists
  // m11 .loaded — 3px green left border, padding adjusted to keep total 1px+12px
  const isLoaded = model.status === 'loaded'

  return (
    <div
      className="rounded-md"
      onContextMenu={onContextMenu}
      style={{
        background: 'var(--card)',
        border: '1px solid var(--border)',
        borderLeft: isLoaded ? '3px solid var(--accent-2)' : '1px solid var(--border)',
        padding: isLoaded ? '10px 12px 10px 10px' : '10px 12px',
        opacity: notDownloaded ? 0.6 : 1,
        cursor: 'context-menu',
        transition: 'border-color 0.15s ease',
      }}
    >
      {/* Row 1: Name + badges + status */}
      <div className="flex items-center gap-2 mb-1">
        <span
          style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-strong)' }}
          className="truncate flex-1"
        >
          {model.organization ? `${model.organization}/` : ''}
          {model.display_name}
        </span>
        <CopyButton text={model.name} />
        {model.auto_detected && (
          <span
            style={{
              fontSize: 8,
              padding: '1px 5px',
              borderRadius: 3,
              background: 'color-mix(in srgb, var(--accent-2) 18%, transparent)',
              color: 'var(--accent-2)',
              flexShrink: 0,
            }}
          >
            自动检测
          </span>
        )}
        {!model.has_adapter && (
          <span
            title="adapter 未实现，无法加载。需先在 backend/configs/models.yaml 添加 adapter 字段。"
            style={{
              fontSize: 8,
              padding: '1px 5px',
              borderRadius: 3,
              background: 'rgba(239,68,68,0.14)',
              color: 'var(--error, #ef4444)',
              flexShrink: 0,
            }}
          >
            未注册
          </span>
        )}
        <StatusBadge
          status={model.status}
          loadedGpus={model.loaded_gpus}
          modelName={model.name}
          detail={model.status_detail}
        />
      </div>

      {/* Row 2: Tags line */}
      <div className="flex flex-wrap items-center gap-1.5" style={{ fontSize: 9, color: 'var(--muted)' }}>
        <Tag
          color={TYPE_TAG_STYLE[model.type]?.color ?? 'var(--accent-2)'}
          bg={TYPE_TAG_STYLE[model.type]?.bg ?? 'var(--accent-2-subtle)'}
        >
          {TYPE_LABELS[model.type]?.split(' ')[0] ?? model.type}
        </Tag>
        {model.model_size && <Tag icon="📦">{model.model_size}</Tag>}
        {/* image engines: surface LoRA count so operator can verify the
         scanner is finding their files without leaving the page. */}
        {model.type === 'image' && model.lora_count !== null && (
          <Tag color="var(--info)">{model.lora_count} LoRA</Tag>
        )}
        {model.frameworks?.map((f) => (
          <Tag key={f} icon="⚙">{f}</Tag>
        ))}
        {model.license && <Tag icon="📄">{model.license}</Tag>}
        {model.languages && model.languages.length > 0 && (
          model.languages.length <= 3
            ? model.languages.map((l) => <Tag key={l}>{l.toUpperCase()}</Tag>)
            : <Tag>{model.languages.length} languages</Tag>
        )}
        {model.tags?.slice(0, 3).map((t) => (
          <span key={t} style={{ color: 'var(--muted)' }}>• {t}</span>
        ))}
      </div>

      {/* Row 3: Local info (read-only) */}
      {/* Row 3a: chips (VRAM / GPU / resident toggle) — m11 style */}
      <div className="flex items-center gap-3 mt-1" style={{ fontSize: 9, color: 'var(--muted)' }}>
        <span>{model.vram_gb}GB VRAM</span>
        <span>GPU {Array.isArray(model.gpu) ? model.gpu.join(',') : model.gpu}</span>
        <button
          title={model.resident ? '点击取消常驻' : '点击设为常驻（不会被自动卸载）'}
          onClick={(e) => {
            e.stopPropagation()
            onToggleResident(model.name, !model.resident)
          }}
          style={{
            color: model.resident ? 'var(--warn)' : 'var(--muted)',
            background: model.resident
              ? 'color-mix(in srgb, var(--warn) 15%, transparent)'
              : 'var(--bg)',
            padding: '1px 5px',
            borderRadius: 3,
            border: 'none',
            cursor: 'pointer',
            fontSize: 9,
          }}
        >
          {model.resident ? 'resident' : 'on-demand'}
        </button>
      </div>

      {/* Row 3b: path on its own line, full width — m11 has `.src { flex: 1 1 100% }` */}
      {model.local_path && (
        <div
          className="mt-1"
          style={{
            fontSize: 9,
            color: model.local_exists ? 'var(--muted-strong, var(--muted))' : 'var(--warn)',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            fontFamily: 'monospace',
          }}
          title={model.local_path}
        >
          <span style={{ color: model.local_exists ? 'var(--accent-2)' : 'var(--warn)', marginRight: 4 }}>
            {model.local_exists ? '✓' : '✗'}
          </span>
          {model.local_path}
        </div>
      )}
    </div>
  )
}

function Tag({
  children,
  icon,
  color,
  bg,
}: {
  children: React.ReactNode
  icon?: string
  color?: string
  bg?: string
}) {
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 2,
        padding: '1px 5px',
        borderRadius: 3,
        background: bg ?? (color ? `color-mix(in srgb, ${color} 12%, transparent)` : 'var(--bg)'),
        color: color ?? 'var(--muted)',
        fontSize: 9,
        whiteSpace: 'nowrap',
      }}
    >
      {icon && <span>{icon}</span>}
      {children}
    </span>
  )
}

// Per-model loading start times (module-scoped so they survive remounts within session)
const _loadingStartedAt: Map<string, number> = new Map()

function StatusBadge({
  status,
  loadedGpus,
  modelName,
  detail,
}: {
  status: string
  loadedGpus?: number[] | null
  modelName?: string
  detail?: string | null
}) {
  const [, force] = useState(0)

  // Track when loading/installing started; tick every second to refresh elapsed
  useEffect(() => {
    if (!modelName) return
    if (status === 'loading' || status === 'installing') {
      if (!_loadingStartedAt.has(modelName)) {
        _loadingStartedAt.set(modelName, Date.now())
      }
      const id = setInterval(() => force((n) => n + 1), 1000)
      return () => clearInterval(id)
    }
    _loadingStartedAt.delete(modelName)
  }, [status, modelName])

  const gpuLabel = loadedGpus && loadedGpus.length > 0
    ? ` · GPU ${loadedGpus.join(',')}`
    : ''

  let elapsedLabel = 'loading...'
  if ((status === 'loading' || status === 'installing') && modelName) {
    const startedAt = _loadingStartedAt.get(modelName)
    if (startedAt) {
      const s = Math.floor((Date.now() - startedAt) / 1000)
      const verb = status === 'installing' ? 'installing' : 'loading'
      elapsedLabel = s < 60 ? `${verb} ${s}s` : `${verb} ${Math.floor(s / 60)}m${s % 60}s`
    } else {
      elapsedLabel = status === 'installing' ? 'installing...' : 'loading...'
    }
  }

  const config: Record<string, { color: string; label: string; animate?: boolean }> = {
    loaded:         { color: 'var(--ok)',           label: `running${gpuLabel}` },
    loading:        { color: 'var(--warn)',         label: elapsedLabel, animate: true },
    failed:         { color: 'var(--accent)',       label: 'failed' },
    installing:     { color: 'var(--warn)',         label: elapsedLabel, animate: true },
    installed:      { color: 'var(--muted-strong)', label: 'installed' },
    install_failed: { color: 'var(--accent)',       label: 'install failed' },
    unloaded:       { color: 'var(--muted-strong)', label: 'idle' },
  }
  const { color, label, animate } = config[status] ?? config.unloaded

  const tooltip = detail || ({
    failed: 'load failed',
    loading: '正在加载',
    installing: '正在安装依赖',
    install_failed: 'dep install failed',
  } as Record<string, string>)[status]

  return (
    <span
      className="flex items-center gap-1"
      style={{ fontSize: 9, color, flexShrink: 0, cursor: tooltip ? 'help' : 'default' }}
      title={tooltip}
    >
      <span
        className="inline-block rounded-full"
        style={{
          width: 6,
          height: 6,
          background: color,
          ...(animate ? { animation: 'loading-pulse 1.5s ease-in-out infinite' } : {}),
        }}
      />
      {label}
    </span>
  )
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      title={`复制: ${text}`}
      onClick={(e) => {
        e.stopPropagation()
        navigator.clipboard.writeText(text)
        setCopied(true)
        setTimeout(() => setCopied(false), 1500)
      }}
      style={{
        background: 'none',
        border: 'none',
        cursor: 'pointer',
        padding: 2,
        color: copied ? 'var(--ok)' : 'var(--muted)',
        flexShrink: 0,
        display: 'flex',
        alignItems: 'center',
      }}
    >
      {copied ? <Check size={12} /> : <Copy size={12} />}
    </button>
  )
}
