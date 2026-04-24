import { useState } from 'react'
import { AlertTriangle, BarChart3, ChevronRight, Cpu, X } from 'lucide-react'
import {
  useSysGpus, useSysStats, useSysProcesses, useKillProcess,
  type SysGpuInfo, type GpuProcessInfo,
} from '../../api/system'
import { useEngines } from '../../api/engines'
import { useDashboardSummary, type AlertItem, type TopServiceRow } from '../../api/dashboard'

/**
 * m04 Dashboard — v3 layout.
 *
 * Top: 4 business stats (today's calls / month tokens / active alerts /
 *   key·service counts).
 * Mid: alerts feed + today's top services.
 * Bottom: collapsible "system status" with GPU panels, mini stats,
 *   loaded models, processes table — keeps the v2 monitoring detail
 *   accessible without dominating the page.
 */
export default function DashboardOverlay() {
  const summary = useDashboardSummary()
  const [sysOpen, setSysOpen] = useState(false)

  return (
    <div
      className="absolute inset-0 overflow-y-auto z-[16]"
      style={{ background: 'var(--bg)' }}
    >
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: 20 }}>
        <div style={{ marginBottom: 14 }}>
          <h1 style={{ fontSize: 20, color: 'var(--text)', fontWeight: 600 }}>概览</h1>
          <p style={{ fontSize: 13, color: 'var(--muted)', marginTop: 4 }}>
            今日业务鸟瞰 + 系统状态
          </p>
        </div>

        {/* Top business stats */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: 12,
            marginBottom: 12,
          }}
        >
          <BizStat
            label="今日调用"
            value={fmtN(summary.data?.today_calls)}
            sub={deltaText(summary.data?.today_calls_delta_pct)}
            subColor={
              summary.data?.today_calls_delta_pct == null
                ? 'var(--muted)'
                : summary.data.today_calls_delta_pct >= 0
                  ? 'var(--accent-2, #22c55e)'
                  : 'var(--warn, #f59e0b)'
            }
          />
          <BizStat
            label="本月 token 用量"
            value={fmtN(summary.data?.month_tokens)}
            sub={
              summary.data?.month_tokens_quota
                ? `占 ${fmtN(summary.data.month_tokens_quota)} 配额 ${summary.data.month_tokens_used_pct}%`
                : '暂未挂资源包'
            }
          />
          <BizStat
            label="活跃告警"
            value={summary.data ? String(summary.data.active_alerts_count) : '—'}
            valueColor={
              (summary.data?.active_alerts_count ?? 0) > 0
                ? 'var(--warn, #f59e0b)'
                : undefined
            }
            sub={summary.data?.active_alerts_top_label ?? '一切正常'}
          />
          <BizStat
            label="API Key · 实例"
            value={
              summary.data
                ? `${summary.data.api_key_count} · ${summary.data.service_count}`
                : '—'
            }
            sub={
              summary.data && summary.data.unbound_key_count > 0
                ? `${summary.data.unbound_key_count} key 未绑定`
                : ''
            }
          />
        </div>

        {/* Alerts feed */}
        <Panel
          icon={<AlertTriangle size={14} style={{ color: 'var(--warn, #f59e0b)' }} />}
          title="活跃告警 · 最近触发"
        >
          {(summary.data?.recent_alerts.length ?? 0) === 0 ? (
            <div style={{ fontSize: 12, color: 'var(--muted)', padding: '8px 0' }}>
              过去 7 天无告警触发
            </div>
          ) : (
            summary.data!.recent_alerts.map((a) => <AlertRow key={a.id} alert={a} />)
          )}
        </Panel>

        {/* Top services today */}
        <Panel
          icon={<BarChart3 size={14} style={{ color: 'var(--accent-2, #22c55e)' }} />}
          title="今日 Top 调用服务"
        >
          {(summary.data?.top_services_today.length ?? 0) === 0 ? (
            <div style={{ fontSize: 12, color: 'var(--muted)', padding: '8px 0' }}>
              今日暂无 LLM 调用
            </div>
          ) : (
            summary.data!.top_services_today.map((s) => (
              <TopSvcStrip key={s.service_name} svc={s} />
            ))
          )}
        </Panel>

        {/* System status — collapsible (was the entire v2 dashboard) */}
        <CollapsibleSystem open={sysOpen} onToggle={() => setSysOpen((v) => !v)} />
      </div>
    </div>
  )
}

// ---------- top-level pieces ----------

function BizStat({
  label,
  value,
  sub,
  valueColor,
  subColor,
}: {
  label: string
  value: string
  sub: string
  valueColor?: string
  subColor?: string
}) {
  return (
    <div
      style={{
        background: 'var(--bg-accent)',
        border: '1px solid var(--border)',
        borderRadius: 8,
        padding: '14px 16px',
      }}
    >
      <div style={{ fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase' }}>
        {label}
      </div>
      <div
        style={{
          fontSize: 24,
          fontWeight: 600,
          color: valueColor ?? 'var(--text)',
          marginTop: 4,
          fontVariantNumeric: 'tabular-nums',
        }}
      >
        {value}
      </div>
      {sub && (
        <div
          style={{ fontSize: 11, color: subColor ?? 'var(--muted)', marginTop: 4 }}
        >
          {sub}
        </div>
      )}
    </div>
  )
}

function Panel({
  icon,
  title,
  rightSlot,
  children,
}: {
  icon: React.ReactNode
  title: string
  rightSlot?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <div
      style={{
        background: 'var(--bg-accent)',
        border: '1px solid var(--border)',
        borderRadius: 8,
        padding: '14px 16px',
        marginBottom: 12,
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          marginBottom: 10,
        }}
      >
        {icon}
        <h3 style={{ fontSize: 13, color: 'var(--text)', fontWeight: 500, flex: 1 }}>
          {title}
        </h3>
        {rightSlot}
      </div>
      {children}
    </div>
  )
}

function AlertRow({ alert }: { alert: AlertItem }) {
  const dotColor =
    alert.severity === 'err' ? 'var(--accent, #ef4444)' : 'var(--warn, #f59e0b)'
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '8px 0',
        borderBottom: '1px solid var(--border)',
        fontSize: 12,
      }}
    >
      <span
        style={{
          width: 6,
          height: 6,
          borderRadius: '50%',
          background: dotColor,
          flexShrink: 0,
        }}
      />
      <div style={{ flex: 1, color: 'var(--text)' }}>
        <strong style={{ color: 'var(--text)' }}>
          {alert.service_name ?? '(unknown)'}
        </strong>{' '}
        <span style={{ color: 'var(--muted)' }}>
          触发 {alert.threshold_percent}% 阈值
        </span>
      </div>
      <span style={{ color: 'var(--muted)', fontSize: 11 }}>
        {fmtTimeAgo(alert.last_notified_at)}
      </span>
    </div>
  )
}

function TopSvcStrip({ svc }: { svc: TopServiceRow }) {
  const pct = Math.max(0, Math.min(100, svc.percent))
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 14,
        padding: '10px 0',
        borderBottom: '1px solid var(--border)',
        fontSize: 12,
      }}
    >
      <div
        style={{
          width: 140,
          color: 'var(--text)',
          fontWeight: 500,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {svc.service_name}
      </div>
      <div
        style={{
          flex: 1,
          height: 6,
          background: 'var(--border)',
          borderRadius: 3,
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: '100%',
            background: 'var(--accent-2, #22c55e)',
          }}
        />
      </div>
      <div
        style={{
          color: 'var(--muted)',
          fontVariantNumeric: 'tabular-nums',
          width: 80,
          textAlign: 'right',
        }}
      >
        {fmtN(svc.calls)} calls
      </div>
      <div
        style={{
          color: 'var(--muted)',
          fontVariantNumeric: 'tabular-nums',
          width: 50,
          textAlign: 'right',
        }}
      >
        {pct.toFixed(0)}%
      </div>
    </div>
  )
}

// ---------- collapsible system status (the v2 dashboard, folded down) ----------

function CollapsibleSystem({
  open,
  onToggle,
}: {
  open: boolean
  onToggle: () => void
}) {
  const { data: gpuData } = useSysGpus()
  const { data: engines } = useEngines()
  const { data: sysStats } = useSysStats()
  const { data: procData } = useSysProcesses()
  const killProcess = useKillProcess()

  const loadedModels = (engines ?? []).filter((e) => e.status === 'loaded')
  const subtitle = `${gpuData?.gpus.length ?? 0}× GPU · ${loadedModels.length} 模型常驻`

  const fmt = (n: number, d = 1) => n.toFixed(d)

  return (
    <div
      style={{
        background: 'var(--bg-accent)',
        border: '1px solid var(--border)',
        borderRadius: 8,
        marginBottom: 12,
        overflow: 'hidden',
      }}
    >
      <button
        type="button"
        onClick={onToggle}
        style={{
          padding: '12px 16px',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          width: '100%',
          background: 'transparent',
          border: 'none',
          borderBottom: open ? '1px solid var(--border)' : 'none',
          cursor: 'pointer',
          color: 'var(--text)',
          textAlign: 'left',
        }}
      >
        <ChevronRight
          size={14}
          style={{
            transform: open ? 'rotate(90deg)' : 'rotate(0)',
            transition: 'transform 0.15s',
          }}
        />
        <Cpu size={14} style={{ color: 'var(--muted)' }} />
        <h3 style={{ fontSize: 13, color: 'var(--text)', fontWeight: 500, flex: 1 }}>
          系统状态
        </h3>
        <span style={{ fontSize: 11, color: 'var(--muted)' }}>{subtitle}</span>
      </button>

      {open && (
        <div style={{ padding: '14px 16px' }}>
          {/* GPU panels */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
              gap: 12,
              marginBottom: 14,
            }}
          >
            {(gpuData?.gpus ?? []).map((gpu) => (
              <GpuCard
                key={gpu.index}
                gpu={gpu}
                onKill={(pid, mem) => {
                  if (
                    window.confirm(
                      `Kill process PID ${pid}? This will free ~${(mem / 1024).toFixed(1)}G GPU memory.`,
                    )
                  ) {
                    killProcess.mutate(pid)
                  }
                }}
              />
            ))}
            {!gpuData && (
              <div style={{ fontSize: 11, color: 'var(--muted)', padding: 16 }}>
                等待 GPU 数据…
              </div>
            )}
          </div>

          {/* mini system stats */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(4, 1fr)',
              gap: 10,
              marginBottom: 14,
            }}
          >
            <MiniStat
              label="CPU"
              value={sysStats ? `${fmt(sysStats.cpu_usage_percent, 0)}%` : '—'}
              hint={sysStats ? `${sysStats.cpu_count} cores` : ''}
            />
            <MiniStat
              label="RAM"
              value={sysStats ? `${fmt(sysStats.memory_used_gb)}G` : '—'}
              hint={sysStats ? `/ ${fmt(sysStats.memory_total_gb)}G` : ''}
            />
            <MiniStat
              label="SWAP"
              value={sysStats ? `${fmt(sysStats.swap_used_gb)}G` : '—'}
              hint={sysStats ? `/ ${fmt(sysStats.swap_total_gb)}G` : ''}
            />
            <MiniStat
              label="DISK"
              value={sysStats ? `${fmt(sysStats.disk_used_gb)}G` : '—'}
              hint={sysStats ? `/ ${fmt(sysStats.disk_total_gb)}G` : ''}
            />
          </div>

          {/* loaded models */}
          <div
            style={{
              fontSize: 11,
              color: 'var(--muted)',
              textTransform: 'uppercase',
              letterSpacing: 0.5,
              margin: '4px 0 6px',
            }}
          >
            已加载模型 ({loadedModels.length})
          </div>
          {loadedModels.length === 0 && (
            <div style={{ fontSize: 12, color: 'var(--muted)', padding: '4px 0' }}>
              无常驻模型
            </div>
          )}
          {loadedModels.map((m) => (
            <div
              key={m.name}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '6px 0',
                fontSize: 12,
              }}
            >
              <span
                style={{
                  width: 7,
                  height: 7,
                  borderRadius: '50%',
                  background: 'var(--ok, #22c55e)',
                }}
              />
              <span style={{ color: 'var(--text)' }}>{m.name}</span>
              <span
                style={{
                  fontSize: 10,
                  padding: '1px 6px',
                  borderRadius: 3,
                  background: 'rgba(34,197,94,0.15)',
                  color: 'var(--accent-2, #22c55e)',
                }}
              >
                {m.type.toUpperCase()}
              </span>
              {m.loaded_gpus && m.loaded_gpus.length > 0 && (
                <span style={{ marginLeft: 'auto', color: 'var(--muted)', fontSize: 11 }}>
                  GPU {m.loaded_gpus.join(',')} · {m.vram_gb}GB
                </span>
              )}
            </div>
          ))}

          {/* process table */}
          <div
            style={{
              fontSize: 11,
              color: 'var(--muted)',
              textTransform: 'uppercase',
              letterSpacing: 0.5,
              margin: '14px 0 6px',
            }}
          >
            进程 ({procData?.processes.length ?? 0})
          </div>
          {procData?.processes ? (
            <div style={{ overflowX: 'auto' }}>
              <table
                style={{
                  width: '100%',
                  borderCollapse: 'collapse',
                  fontSize: 11,
                  fontFamily: 'var(--mono, monospace)',
                }}
              >
                <thead>
                  <tr style={{ color: 'var(--muted)', textAlign: 'left' }}>
                    <th style={{ padding: '4px 8px' }}>PID</th>
                    <th style={{ padding: '4px 8px' }}>CPU%</th>
                    <th style={{ padding: '4px 8px' }}>MEM</th>
                    <th style={{ padding: '4px 8px' }}>NAME</th>
                    <th style={{ padding: '4px 8px' }}>COMMAND</th>
                  </tr>
                </thead>
                <tbody>
                  {procData.processes.slice(0, 15).map((p) => (
                    <tr key={p.pid} style={{ borderTop: '1px solid var(--border)' }}>
                      <td style={{ padding: '3px 8px', color: 'var(--muted)' }}>{p.pid}</td>
                      <td style={{ padding: '3px 8px', color: 'var(--muted)' }}>
                        {p.cpu_percent.toFixed(1)}%
                      </td>
                      <td style={{ padding: '3px 8px', color: 'var(--warn, #f59e0b)' }}>
                        {p.memory_mb}M
                      </td>
                      <td style={{ padding: '3px 8px', color: 'var(--muted)' }}>{p.name}</td>
                      <td
                        style={{
                          padding: '3px 8px',
                          color: 'var(--accent-2, #22c55e)',
                          maxWidth: 600,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {p.command}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div style={{ fontSize: 11, color: 'var(--muted)', padding: 8 }}>
              启动 nous-center-sys 以获取进程数据
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function MiniStat({
  label,
  value,
  hint,
}: {
  label: string
  value: string
  hint?: string
}) {
  return (
    <div
      style={{
        background: 'var(--bg)',
        padding: '10px 12px',
        borderRadius: 4,
      }}
    >
      <div style={{ fontSize: 10, color: 'var(--muted)', textTransform: 'uppercase' }}>
        {label}
      </div>
      <div
        style={{
          fontSize: 14,
          fontWeight: 500,
          color: 'var(--text)',
          fontVariantNumeric: 'tabular-nums',
          marginTop: 2,
        }}
      >
        {value}
        {hint && (
          <span style={{ color: 'var(--muted)', fontSize: 10, marginLeft: 6 }}>
            {hint}
          </span>
        )}
      </div>
    </div>
  )
}

function GpuCard({
  gpu,
  onKill,
}: {
  gpu: SysGpuInfo
  onKill: (pid: number, mem: number) => void
}) {
  const memPct =
    gpu.memory_total_mb > 0 ? (gpu.memory_used_mb / gpu.memory_total_mb) * 100 : 0
  return (
    <div
      style={{
        background: 'var(--bg)',
        border: '1px solid var(--border)',
        borderRadius: 6,
        padding: '12px 14px',
      }}
    >
      <div
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
      >
        <span
          style={{
            fontSize: 11,
            fontWeight: 600,
            color: 'var(--accent-2, #22c55e)',
            letterSpacing: 0.5,
          }}
        >
          GPU {gpu.index}
        </span>
        <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--text)' }}>
          {gpu.utilization_gpu}%
        </span>
      </div>
      <div style={{ fontSize: 13, color: 'var(--text)', marginTop: 4 }}>{gpu.name}</div>
      <div
        style={{
          fontSize: 11,
          color: 'var(--muted)',
          display: 'flex',
          gap: 12,
          marginTop: 4,
          marginBottom: 10,
        }}
      >
        <span style={{ color: 'var(--accent-2, #22c55e)' }}>{gpu.temperature}°C</span>
        <span>FAN {gpu.fan_speed}%</span>
        <span>POW {gpu.power_draw_w.toFixed(0)}/{gpu.power_limit_w.toFixed(0)}W</span>
      </div>
      <BarRow label="GPU" value={`${gpu.utilization_gpu}%`} pct={gpu.utilization_gpu} />
      <BarRow
        label="MEM"
        value={`${(gpu.memory_used_mb / 1024).toFixed(1)}G / ${(gpu.memory_total_mb / 1024).toFixed(0)}G`}
        pct={memPct}
        warn={gpu.low_memory}
      />
      {gpu.processes && gpu.processes.length > 0 && (
        <div
          style={{ marginTop: 10, paddingTop: 10, borderTop: '1px dashed var(--border)' }}
        >
          <div
            style={{
              fontSize: 9,
              fontWeight: 600,
              color: 'var(--muted)',
              textTransform: 'uppercase',
              letterSpacing: 0.5,
              marginBottom: 6,
            }}
          >
            GPU Processes
          </div>
          {gpu.processes.map((proc) => (
            <ProcRow key={proc.pid} proc={proc} onKill={onKill} />
          ))}
        </div>
      )}
    </div>
  )
}

function BarRow({
  label,
  value,
  pct,
  warn,
}: {
  label: string
  value: string
  pct: number
  warn?: boolean
}) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '4px 0',
        fontSize: 11,
      }}
    >
      <span
        style={{
          width: 32,
          fontSize: 10,
          color: 'var(--muted)',
          textTransform: 'uppercase',
        }}
      >
        {label}
      </span>
      <div
        style={{
          flex: 1,
          height: 4,
          background: 'var(--border)',
          borderRadius: 2,
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${Math.min(100, pct)}%`,
            height: '100%',
            background: warn ? 'var(--warn, #f59e0b)' : 'var(--accent-2, #22c55e)',
          }}
        />
      </div>
      <span
        style={{
          color: 'var(--muted)',
          fontVariantNumeric: 'tabular-nums',
          minWidth: 70,
          textAlign: 'right',
        }}
      >
        {value}
      </span>
    </div>
  )
}

function ProcRow({
  proc,
  onKill,
}: {
  proc: GpuProcessInfo
  onKill: (pid: number, mem: number) => void
}) {
  const memG = (proc.used_gpu_memory_mb / 1024).toFixed(1)
  const isOrphan = !proc.managed
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '3px 0',
        fontSize: 11,
        fontFamily: 'var(--mono, monospace)',
        color: isOrphan ? 'var(--warn, #f59e0b)' : 'var(--muted)',
      }}
    >
      <span style={{ minWidth: 48 }}>{proc.pid}</span>
      <span style={{ minWidth: 44 }}>{memG}G</span>
      <span
        style={{
          flex: 1,
          color: 'var(--text)',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {proc.managed ? proc.model_name : proc.command}
      </span>
      {proc.managed ? (
        <span
          style={{
            fontSize: 9,
            padding: '1px 5px',
            borderRadius: 3,
            background: 'rgba(34,197,94,0.15)',
            color: 'var(--accent-2, #22c55e)',
          }}
        >
          managed
        </span>
      ) : (
        <>
          <span
            style={{
              fontSize: 9,
              padding: '1px 5px',
              borderRadius: 3,
              background: 'var(--warn, #f59e0b)',
              color: '#1a1a1a',
              fontFamily: 'inherit',
            }}
          >
            orphan
          </span>
          <button
            type="button"
            onClick={() => onKill(proc.pid, proc.used_gpu_memory_mb)}
            title={`Kill ${proc.pid}`}
            style={{
              width: 18,
              height: 18,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              borderRadius: 3,
              background: 'rgba(99,102,241,0.15)',
              color: 'var(--accent)',
              border: 'none',
              cursor: 'pointer',
            }}
          >
            <X size={10} />
          </button>
        </>
      )}
    </div>
  )
}

// ---------- formatters ----------

function fmtN(n: number | undefined): string {
  if (n == null) return '—'
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 10_000) return `${(n / 1_000).toFixed(1)}K`
  return n.toLocaleString()
}

function deltaText(pct: number | null | undefined): string {
  if (pct == null) return '昨日无数据'
  const arrow = pct >= 0 ? '↑' : '↓'
  return `${arrow} ${Math.abs(pct).toFixed(1)}% vs 昨天`
}

function fmtTimeAgo(iso: string | null): string {
  if (!iso) return ''
  const d = new Date(iso)
  const diffSec = (Date.now() - d.getTime()) / 1000
  if (diffSec < 60) return `${Math.floor(diffSec)} 秒前`
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)} 分钟前`
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)} 小时前`
  return `${Math.floor(diffSec / 86400)} 天前`
}
