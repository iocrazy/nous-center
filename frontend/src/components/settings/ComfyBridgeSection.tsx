import { useState } from 'react'
import { AlertCircle } from 'lucide-react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { getComfyHealth, freeComfyVram, type ComfyDevice } from '../../api/comfyTemplates'

const GB = 1_000_000_000

function formatGB(bytes: number): string {
  return `${(bytes / GB).toFixed(1)}GB`
}

// "cuda:0 NVIDIA RTX PRO 6000 ... : cudaMallocAsync" -> "NVIDIA RTX PRO 6000 ..."
function formatDeviceName(name: string): string {
  return name.replace(/^\S+:\d+\s+/, '').replace(/\s*:\s*\S+$/, '')
}

function barColor(ratio: number): string {
  if (ratio >= 0.9) return 'var(--accent)'
  if (ratio >= 0.7) return 'var(--warn)'
  return 'var(--ok)'
}

function DeviceRow({ device }: { device: ComfyDevice }) {
  const ratio = device.vram_total > 0 ? device.vram_used / device.vram_total : 0
  return (
    <div style={{ padding: '10px 0', borderTop: '1px solid var(--border)' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: 12,
          color: 'var(--text)',
          marginBottom: 6,
        }}
      >
        <span>{formatDeviceName(device.name)}</span>
        <span style={{ color: 'var(--muted)', fontFamily: 'var(--mono, monospace)' }}>
          {formatGB(device.vram_used)} / {formatGB(device.vram_total)}
        </span>
      </div>
      <div
        style={{
          height: 6,
          borderRadius: 3,
          background: 'var(--bg)',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            height: '100%',
            width: `${Math.min(100, Math.max(0, ratio * 100))}%`,
            background: barColor(ratio),
            transition: 'width 0.3s ease',
          }}
        />
      </div>
    </div>
  )
}

export default function ComfyBridgeSection() {
  const queryClient = useQueryClient()
  const [freeing, setFreeing] = useState(false)
  const [freeError, setFreeError] = useState<string | null>(null)

  const { data: health, isLoading, isError } = useQuery({
    queryKey: ['comfy-health'],
    queryFn: getComfyHealth,
    retry: false,
    refetchOnWindowFocus: false,
  })

  const formatTimeout = (seconds: number): string => {
    if (seconds >= 3600) {
      const hours = seconds / 3600
      if (Number.isInteger(hours)) {
        return `${hours}h`
      }
      return `${hours.toFixed(1)}h`
    }
    return `${seconds}s`
  }

  const handleFree = async () => {
    setFreeing(true)
    setFreeError(null)
    try {
      await freeComfyVram()
      await queryClient.invalidateQueries({ queryKey: ['comfy-health'] })
    } catch (e) {
      setFreeError(e instanceof Error ? e.message : '释放显存失败')
    } finally {
      setFreeing(false)
    }
  }

  const devices = health?.devices ?? []

  return (
    <div>
      <div style={{ marginBottom: 18 }}>
        <h2 style={{ fontSize: 16, fontWeight: 600, color: 'var(--text)', marginBottom: 4 }}>
          ComfyUI 桥
        </h2>
        <div style={{ fontSize: 12, color: 'var(--muted)' }}>
          ComfyUI 工作流编排 - 网络连接与队列状态
        </div>
      </div>

      {isLoading && (
        <div
          style={{
            padding: 16,
            fontSize: 12,
            color: 'var(--muted)',
            textAlign: 'center',
          }}
        >
          获取状态中...
        </div>
      )}

      {isError && (
        <div
          style={{
            marginTop: 18,
            padding: '12px 14px',
            background: 'rgba(239,68,68,0.08)',
            borderLeft: '3px solid var(--error, #ef4444)',
            borderRadius: '0 4px 4px 0',
            fontSize: 12,
            color: 'var(--text)',
            display: 'flex',
            alignItems: 'center',
            gap: 8,
          }}
        >
          <AlertCircle size={14} style={{ flexShrink: 0 }} />
          无法获取状态
        </div>
      )}

      {health && !isError && (
        <>
          <div
            style={{
              marginTop: 18,
              padding: 16,
              background: 'var(--bg-accent)',
              border: '1px solid var(--border)',
              borderRadius: 6,
            }}
          >
            {/* Status Row */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                marginBottom: 12,
              }}
            >
              <div
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  background: health.online
                    ? 'var(--ok, #22c55e)'
                    : 'var(--warn, #f59e0b)',
                  flexShrink: 0,
                }}
              />
              <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text)' }}>
                {health.online
                  ? `在线 · 队列 ${health.queue_depth} · ComfyUI ${health.version}`
                  : '离线 — 检查 sidecar 服务 (enginectl status)'}
              </span>
            </div>

            {/* Details Row */}
            {health.online && (
              <>
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '160px 1fr',
                    gap: 16,
                    alignItems: 'center',
                    paddingTop: 12,
                    borderTop: '1px solid var(--border)',
                  }}
                >
                  <span style={{ fontSize: 12, color: 'var(--muted)' }}>服务地址</span>
                  <code
                    style={{
                      fontSize: 12,
                      color: 'var(--text)',
                      fontFamily: 'var(--mono, monospace)',
                      background: 'var(--bg)',
                      padding: '4px 8px',
                      borderRadius: 3,
                      display: 'inline-block',
                      maxWidth: '100%',
                      overflowX: 'auto',
                    }}
                  >
                    {health.base_url}
                  </code>
                </div>

                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '160px 1fr',
                    gap: 16,
                    alignItems: 'center',
                    paddingTop: 12,
                    borderTop: '1px solid var(--border)',
                  }}
                >
                  <span style={{ fontSize: 12, color: 'var(--muted)' }}>任务超时</span>
                  <span style={{ fontSize: 13, color: 'var(--text)' }}>
                    超时 {formatTimeout(health.timeout_s)}
                  </span>
                </div>
              </>
            )}

            {/* GPU VRAM */}
            {health.online && devices.length > 0 && (
              <div style={{ marginTop: 4 }}>
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    paddingTop: 12,
                    borderTop: '1px solid var(--border)',
                  }}
                >
                  <span style={{ fontSize: 12, color: 'var(--muted)' }}>显存占用</span>
                  <button
                    onClick={handleFree}
                    disabled={freeing}
                    style={{
                      fontSize: 12,
                      fontWeight: 500,
                      padding: '5px 10px',
                      borderRadius: 4,
                      border: '1px solid var(--border)',
                      background: 'var(--bg)',
                      color: 'var(--text)',
                      cursor: freeing ? 'default' : 'pointer',
                      opacity: freeing ? 0.6 : 1,
                    }}
                  >
                    {freeing ? '释放中…' : '释放显存'}
                  </button>
                </div>
                {devices.map((d) => (
                  <DeviceRow key={`${d.type}:${d.index}`} device={d} />
                ))}
                {freeError && (
                  <div style={{ marginTop: 6, fontSize: 12, color: 'var(--accent)' }}>
                    {freeError}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Config Note */}
          <div
            style={{
              marginTop: 18,
              padding: '12px 14px',
              background: 'rgba(59,130,246,0.08)',
              borderLeft: '3px solid var(--info, #3b82f6)',
              borderRadius: '0 4px 4px 0',
              fontSize: 12,
              color: 'var(--text)',
              lineHeight: 1.7,
            }}
          >
            地址与超时经 <code style={{ fontFamily: 'var(--mono, monospace)' }}>backend/.env</code> 的{' '}
            <code style={{ fontFamily: 'var(--mono, monospace)' }}>NOUS_COMFY_URL</code> /{' '}
            <code style={{ fontFamily: 'var(--mono, monospace)' }}>NOUS_COMFY_TIMEOUT</code>{' '}
            配置，修改后重启生效。
          </div>
        </>
      )}
    </div>
  )
}
