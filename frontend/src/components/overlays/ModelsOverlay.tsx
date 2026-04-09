import { useState, useCallback } from 'react'
import { Copy, Check } from 'lucide-react'
import {
  useEngines, useLoadEngine, useUnloadEngine, useSyncMetadata,
  useScanModels, useSetResident, useRefreshMetadata, useGpus, useSetGpu,
  type EngineInfo,
} from '../../api/engines'
import ContextMenu, { type MenuItem } from '../ui/ContextMenu'

const TYPE_LABELS: Record<string, string> = {
  llm: '语言模型 LLM',
  tts: '语音合成 TTS',
  image: '图像生成 Image',
  video: '视频生成 Video',
  understand: '多模态理解 VL',
}

const TYPE_ORDER = ['llm', 'tts', 'image', 'video', 'understand']

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

  const closeMenu = useCallback(() => {
    setCtxMenu((prev) => ({ ...prev, visible: false }))
  }, [])

  const handleContextMenu = useCallback((e: React.MouseEvent, model: EngineInfo) => {
    e.preventDefault()
    setCtxMenu({ visible: true, position: { x: e.clientX, y: e.clientY }, model })
  }, [])

  const handleToggle = useCallback(
    (engine: EngineInfo) => {
      if (engine.status === 'loaded') {
        unloadEngine.mutate(engine.name)
      } else {
        loadEngine.mutate(engine.name)
      }
    },
    [loadEngine, unloadEngine],
  )

  // Group by type
  const groups = (engines ?? []).reduce<Record<string, EngineInfo[]>>((acc, e) => {
    ;(acc[e.type] ??= []).push(e)
    return acc
  }, {})

  const hasAnyMissing = (engines ?? []).some((e) => !e.has_metadata)

  // Build context menu items for the active model
  const menuItems: MenuItem[] = ctxMenu.model
    ? [
        {
          label: ctxMenu.model.status === 'loaded' ? '卸载模型' : '加载模型',
          onClick: () => handleToggle(ctxMenu.model!),
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

  return (
    <div
      className="absolute inset-0 overflow-y-auto z-[16]"
      style={{ background: 'var(--bg)' }}
    >
      <div style={{ padding: 16 }}>
        {/* Header with scan + sync buttons */}
        <div className="flex items-center gap-2 mb-3">
          <button
            onClick={() => scanModels.mutate()}
            disabled={scanModels.isPending}
            style={{
              padding: '4px 12px',
              fontSize: 10,
              borderRadius: 4,
              border: '1px solid var(--border)',
              background: 'var(--accent-2)',
              color: '#fff',
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
                padding: '4px 12px',
                fontSize: 10,
                borderRadius: 4,
                border: '1px solid var(--border)',
                background: 'var(--accent)',
                color: '#fff',
                cursor: syncMeta.isPending ? 'wait' : 'pointer',
                opacity: syncMeta.isPending ? 0.6 : 1,
              }}
            >
              {syncMeta.isPending ? '同步中...' : '拉取模型信息'}
            </button>
          )}
          <span style={{ fontSize: 9, color: 'var(--muted)' }}>
            扫描本地目录自动检测模型 · 右键点击卡片可操作
          </span>
        </div>

        {isLoading && (
          <div style={{ fontSize: 11, color: 'var(--muted)' }}>加载中...</div>
        )}

        {isError && !engines && (
          <div style={{ fontSize: 11, color: 'var(--muted)' }}>
            无法连接后端服务，等待重试...
          </div>
        )}

        {TYPE_ORDER.filter((t) => groups[t]).map((type) => (
          <div key={type} className="mb-5">
            <div
              className="flex items-center gap-2 mb-2"
              style={{
                fontSize: 10,
                fontWeight: 600,
                color: 'var(--accent-2)',
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
              }}
            >
              {TYPE_LABELS[type] ?? type}
              <span
                style={{
                  fontSize: 9,
                  color: 'var(--muted)',
                  background: 'var(--card)',
                  padding: '1px 6px',
                  borderRadius: 8,
                }}
              >
                {groups[type].length}
              </span>
            </div>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))',
                gap: 8,
              }}
            >
              {groups[type].map((model) => (
                <ModelCard
                  key={model.name}
                  model={model}
                  onContextMenu={(e) => handleContextMenu(e, model)}
                  onToggleResident={(name, resident) => setResident.mutate({ name, resident })}
                />
              ))}
            </div>
          </div>
        ))}
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

  return (
    <div
      className="rounded-md"
      onContextMenu={onContextMenu}
      style={{
        background: 'var(--card)',
        border: '1px solid var(--border)',
        padding: '10px 12px',
        opacity: notDownloaded ? 0.6 : 1,
        cursor: 'context-menu',
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
        <StatusBadge status={model.status} loadedGpus={model.loaded_gpus} />
      </div>

      {/* Row 2: Tags line */}
      <div className="flex flex-wrap items-center gap-1.5" style={{ fontSize: 9, color: 'var(--muted)' }}>
        <Tag color="var(--accent-2)">{TYPE_LABELS[model.type]?.split(' ')[0] ?? model.type}</Tag>
        {model.model_size && <Tag icon="📦">{model.model_size}</Tag>}
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
        {model.local_path && (
          <span
            style={{ color: model.local_exists ? 'var(--ok)' : 'var(--warn)' }}
            title={model.local_path}
          >
            {model.local_exists ? '✓ ' : '✗ '}
            <span style={{ fontFamily: 'monospace' }}>{model.local_path}</span>
          </span>
        )}
      </div>
    </div>
  )
}

function Tag({
  children,
  icon,
  color,
}: {
  children: React.ReactNode
  icon?: string
  color?: string
}) {
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 2,
        padding: '1px 5px',
        borderRadius: 3,
        background: color ? `color-mix(in srgb, ${color} 12%, transparent)` : 'var(--bg)',
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

function StatusBadge({ status, loadedGpus }: { status: string; loadedGpus?: number[] | null }) {
  const isLoaded = status === 'loaded'
  const gpuLabel = loadedGpus && loadedGpus.length > 0
    ? ` · GPU ${loadedGpus.join(',')}`
    : ''
  return (
    <span
      className="flex items-center gap-1"
      style={{ fontSize: 9, color: isLoaded ? 'var(--ok)' : 'var(--muted-strong)', flexShrink: 0 }}
    >
      <span
        className="inline-block rounded-full"
        style={{
          width: 6,
          height: 6,
          background: isLoaded ? 'var(--ok)' : 'var(--muted-strong)',
        }}
      />
      {isLoaded ? `running${gpuLabel}` : 'idle'}
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
