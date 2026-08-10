import { useEffect, useState } from 'react'
import { AlertCircle } from 'lucide-react'
import { getComfyHealth, type ComfyHealth } from '../../api/comfyTemplates'

export default function ComfyBridgeSection() {
  const [health, setHealth] = useState<ComfyHealth | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const result = await getComfyHealth()
        setHealth(result)
        setError(null)
      } catch (err) {
        setError('无法获取状态')
        setHealth(null)
      } finally {
        setLoading(false)
      }
    }

    fetchHealth()
  }, [])

  const formatTimeout = (seconds: number): string => {
    if (seconds >= 3600) {
      const hours = Math.round(seconds / 3600)
      return `${hours}h`
    }
    return `${seconds}s`
  }

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

      {loading && (
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

      {error && (
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
          {error}
        </div>
      )}

      {health && !error && (
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
