import { X } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts'
import { useSysGpus, useSysStats, useSysProcesses, useMonitorStats, useKillProcess, useUsageSummary, useInferenceUsage, type SysGpuInfo, type GpuProcessInfo } from '../../api/system'
import { useEngines } from '../../api/engines'

export default function DashboardOverlay() {
  const { data: gpuData } = useSysGpus()
  const { data: engines } = useEngines()
  const { data: sysStats } = useSysStats()
  const { data: procData } = useSysProcesses()
  const { data: monitorData } = useMonitorStats()
  const { data: usageData } = useUsageSummary()
  const killProcess = useKillProcess()

  const fmt = (n: number, d = 1) => n.toFixed(d)
  const formatUptime = (s: number) => {
    const d = Math.floor(s / 86400)
    const h = Math.floor((s % 86400) / 3600)
    const m = Math.floor((s % 3600) / 60)
    return d > 0 ? `${d}d ${h}h` : `${h}h ${m}m`
  }

  return (
    <div
      className="absolute inset-0 overflow-y-auto z-[16]"
      style={{ background: 'var(--bg)' }}
    >
      <div style={{ maxWidth: 1400, margin: '0 auto', padding: 20 }}>
        {/* Stats row */}
        <div className="grid grid-cols-4 gap-3 mb-3">
          <StatCard label="API Keys" value="3" sub="2 active" />
          <StatCard label="Today Calls" value={usageData ? String(usageData.today.total_calls) : '--'} sub={usageData ? `LLM ${usageData.today.llm_calls} / TTS ${usageData.today.tts_calls}` : '--'} />
          <StatCard label="Uptime" value={monitorData ? formatUptime(monitorData.uptime_seconds) : '--'} sub="" />
          <StatCard label="Token Usage" value={usageData ? (usageData.today.llm_total_tokens > 1000 ? `${(usageData.today.llm_total_tokens / 1000).toFixed(1)}K` : String(usageData.today.llm_total_tokens)) : '--'} sub={usageData ? `total ${(usageData.all_time.llm_total_tokens / 1000).toFixed(0)}K` : '--'} />
        </div>

        {/* GPU panels */}
        <div className="grid grid-cols-2 gap-3 mb-3">
          {gpuData?.gpus ? (
            gpuData.gpus.map((gpu, i) => (
              <GpuPanel
                key={gpu.index}
                gpu={gpu}
                chartColor={i === 0 ? 'var(--ok)' : 'var(--accent-2)'}
                onKillProcess={(pid, mem) => {
                  if (window.confirm(`Kill process PID ${pid}? This will free ~${(mem / 1024).toFixed(1)}G GPU memory.`)) {
                    killProcess.mutate(pid)
                  }
                }}
              />
            ))
          ) : (
            <>
              <PlaceholderGpuPanel index={0} color="var(--ok)" />
              <PlaceholderGpuPanel index={1} color="var(--accent-2)" />
            </>
          )}
        </div>

        {/* System stats */}
        <div
          className="grid gap-3 mb-3"
          style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}
        >
          <SystemStatCard
            label="CPU"
            value={sysStats ? `${fmt(sysStats.cpu_usage_percent, 0)}%` : '--'}
            sub={sysStats ? `${sysStats.cpu_count} cores` : '--'}
          />
          <SystemStatCard
            label="RAM"
            value={sysStats ? `${fmt(sysStats.memory_used_gb)}G` : '--'}
            sub={sysStats ? `/ ${fmt(sysStats.memory_total_gb)}G` : '--'}
          />
          <SystemStatCard
            label="SWAP"
            value={sysStats ? `${fmt(sysStats.swap_used_gb)}G` : '--'}
            sub={sysStats ? `/ ${fmt(sysStats.swap_total_gb)}G` : '--'}
          />
          <SystemStatCard
            label="Disk"
            value={sysStats ? `${fmt(sysStats.disk_used_gb)}G` : '--'}
            sub={sysStats ? `/ ${fmt(sysStats.disk_total_gb)}G` : '--'}
          />
        </div>

        {/* Loaded Models */}
        {(() => {
          const loadedModels = engines?.filter((e) => e.status === 'loaded') ?? []
          return (
            <MonPanel title={`Loaded Models (${loadedModels.length})`}>
              {loadedModels.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {loadedModels.map((m) => (
                    <div
                      key={m.name}
                      className="flex items-center gap-3"
                      style={{
                        padding: '8px 12px',
                        background: 'var(--card)',
                        borderRadius: 6,
                        border: '1px solid var(--border)',
                      }}
                    >
                      <span
                        style={{
                          width: 8, height: 8, borderRadius: '50%',
                          background: 'var(--ok)', flexShrink: 0,
                        }}
                      />
                      <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-strong)', flex: 1 }}>
                        {m.display_name}
                      </span>
                      <span style={{
                        fontSize: 9, padding: '2px 6px', borderRadius: 3,
                        background: 'color-mix(in srgb, var(--accent-2) 15%, transparent)',
                        color: 'var(--accent-2)',
                      }}>
                        {m.type.toUpperCase()}
                      </span>
                      {m.loaded_gpus && m.loaded_gpus.length > 0 && (
                        <span style={{ fontSize: 9, color: 'var(--muted)' }}>
                          GPU {m.loaded_gpus.join(',')}
                        </span>
                      )}
                      <span style={{ fontSize: 9, color: 'var(--muted)' }}>
                        {m.vram_gb}GB
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ fontSize: 11, color: 'var(--muted)', padding: '12px 0', textAlign: 'center' }}>
                  暂无加载的模型
                </div>
              )}
            </MonPanel>
          )
        })()}

        {/* Process table */}
        <MonPanel title="Processes">
          <div
            className="grid"
            style={{
              gridTemplateColumns: '60px 60px 60px 80px 1fr',
              gap: 4,
              fontSize: 12,
              color: 'var(--accent-2)',
              fontFamily: 'var(--mono)',
              padding: '4px 0',
              borderBottom: '1px solid var(--border)',
              fontWeight: 600,
            }}
          >
            <span>PID</span><span>CPU%</span><span>MEM</span><span>NAME</span><span>COMMAND</span>
          </div>
          {procData?.processes ? (
            procData.processes.slice(0, 15).map((p, i) => (
              <ProcRow
                key={p.pid}
                index={i}
                pid={String(p.pid)}
                cpu={`${p.cpu_percent.toFixed(1)}%`}
                mem={`${p.memory_mb}M`}
                name={p.name}
                cmd={p.command}
              />
            ))
          ) : (
            <div style={{ fontSize: 11, color: 'var(--muted)', padding: '12px 0', textAlign: 'center' }}>
              启动 nous-center-sys 以获取进程数据
            </div>
          )}
        </MonPanel>
      </div>

      {/* Inference usage (Ark-style) */}
      <div className="grid grid-cols-1 gap-3 mt-3">
        <UsageChartCard />
      </div>
    </div>
  )
}

function UsageChartCard() {
  const { data } = useInferenceUsage({ interval: 'day', group_by: 'Model' })
  const rows = (data?.data ?? []).map(r => ({
    day: (r.day ?? '').slice(5, 10),
    model: r.model ?? 'unknown',
    input: r.input_tokens,
    output: r.output_tokens,
    calls: r.req_cnt,
  }))
  return (
    <div
      className="rounded-md"
      style={{ background: 'var(--card)', border: '1px solid var(--border)', padding: 14 }}
    >
      <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--accent-2)', marginBottom: 10 }}>
        近 7 日推理用量（按模型 · 天粒度）
      </div>
      {rows.length === 0 ? (
        <div style={{ fontSize: 11, color: 'var(--muted)', padding: '20px 0', textAlign: 'center' }}>
          暂无数据
        </div>
      ) : (
        <div style={{ width: '100%', height: 260 }}>
          <ResponsiveContainer>
            <BarChart data={rows} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="day" stroke="var(--muted)" fontSize={11} />
              <YAxis stroke="var(--muted)" fontSize={11} />
              <Tooltip
                contentStyle={{
                  background: 'var(--card)',
                  border: '1px solid var(--border)',
                  fontSize: 11,
                }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="input" stackId="t" fill="var(--accent-2)" name="Input tokens" />
              <Bar dataKey="output" stackId="t" fill="var(--accent)" name="Output tokens" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

function StatCard({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div
      className="text-center rounded-md"
      style={{
        background: 'var(--card)',
        border: '1px solid var(--border)',
        padding: '12px 14px',
      }}
    >
      <div style={{ fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ fontSize: 28, fontWeight: 700, color: 'var(--text-strong)', fontFamily: 'var(--mono)' }}>
        {value}
      </div>
      <div style={{ fontSize: 11, color: 'var(--ok)', marginTop: 2 }}>{sub}</div>
    </div>
  )
}

function SystemStatCard({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div
      className="text-center rounded-md"
      style={{
        background: 'var(--card)',
        border: '1px solid var(--border)',
        padding: '12px 14px',
      }}
    >
      <div style={{ fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ fontSize: 24, fontWeight: 700, color: 'var(--text-strong)', fontFamily: 'var(--mono)' }}>
        {value}
      </div>
      <div style={{ fontSize: 11, color: 'var(--ok)', marginTop: 2 }}>{sub}</div>
    </div>
  )
}

function GpuPanel({ gpu, chartColor, onKillProcess }: { gpu: SysGpuInfo; chartColor: string; onKillProcess: (pid: number, memMb: number) => void }) {
  const memPct = gpu.memory_total_mb > 0 ? (gpu.memory_used_mb / gpu.memory_total_mb) * 100 : 0
  const memUsedG = (gpu.memory_used_mb / 1024).toFixed(1)
  const memTotalG = (gpu.memory_total_mb / 1024).toFixed(0)
  const borderColor = gpu.low_memory ? '#ef4444' : chartColor
  const memBarColor = gpu.low_memory ? '#ef4444' : chartColor

  return (
    <MonPanel title={`GPU ${gpu.index}`} titleColor={chartColor} value={`${gpu.utilization_gpu}%`} accentColor={borderColor}>
      {gpu.low_memory && (
        <div style={{
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: 4,
          padding: '4px 8px',
          marginBottom: 8,
          fontSize: 11,
          color: '#ef4444',
          fontWeight: 600,
        }}>
          WARNING: Low VRAM — auto-eviction active
        </div>
      )}
      <div style={{ fontSize: 12, color: 'var(--muted-strong)', fontFamily: 'var(--mono)', lineHeight: 1.8, marginBottom: 8 }}>
        <span style={{ color: 'var(--text)', fontSize: 14, fontWeight: 600 }}>{gpu.name}</span>
        <br />
        <span style={{ color: gpu.temperature < 50 ? 'var(--ok)' : gpu.temperature < 80 ? 'var(--warn)' : '#ef4444' }}>
          {gpu.temperature}°C
        </span> &nbsp;
        <span style={{ color: 'var(--muted)' }}>FAN</span> {gpu.fan_speed}% &nbsp;
        <span style={{ color: 'var(--muted)' }}>POW</span> {gpu.power_draw_w.toFixed(0)}/{gpu.power_limit_w.toFixed(0)}W
      </div>

      {/* GPU utilization bar */}
      <div className="flex items-center gap-2 mb-2">
        <span style={{ fontSize: 12, color: 'var(--muted)', width: 36, fontFamily: 'var(--mono)', fontWeight: 600 }}>GPU</span>
        <div className="flex-1 overflow-hidden" style={{ height: 10, background: 'var(--bg)', borderRadius: 3 }}>
          <div
            style={{
              height: '100%',
              width: `${gpu.utilization_gpu}%`,
              background: chartColor,
              borderRadius: 3,
              transition: 'width 0.5s',
            }}
          />
        </div>
        <span style={{ fontSize: 12, color: 'var(--muted)', fontFamily: 'var(--mono)', width: 36, textAlign: 'right' }}>
          {gpu.utilization_gpu}%
        </span>
      </div>

      {/* Separator */}
      <div style={{ height: 1, background: 'var(--border)', margin: '4px 0 8px' }} />

      {/* Memory bar */}
      <div className="flex items-center gap-2">
        <span style={{ fontSize: 12, color: 'var(--muted)', width: 36, fontFamily: 'var(--mono)', fontWeight: 600 }}>MEM</span>
        <div className="flex-1 overflow-hidden" style={{ height: 10, background: 'var(--bg)', borderRadius: 3 }}>
          <div
            style={{
              height: '100%',
              width: `${memPct}%`,
              background: memBarColor,
              borderRadius: 3,
              transition: 'width 0.5s',
            }}
          />
        </div>
        <span style={{ fontSize: 12, color: gpu.low_memory ? '#ef4444' : 'var(--muted)', fontFamily: 'var(--mono)', width: 80, textAlign: 'right' }}>
          {memUsedG}G / {memTotalG}G
        </span>
      </div>

      {gpu.loaded_models && gpu.loaded_models.length > 0 && (
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {gpu.loaded_models.map((m) => (
            <span key={m.name} style={{
              background: `color-mix(in srgb, ${chartColor} 15%, transparent)`,
              color: chartColor,
              padding: '2px 8px',
              borderRadius: 4,
              fontSize: 11,
              fontWeight: 500,
              border: `1px solid color-mix(in srgb, ${chartColor} 25%, transparent)`,
            }}>
              {m.name} ({m.vram_gb}GB)
            </span>
          ))}
        </div>
      )}

      {gpu.processes && gpu.processes.length > 0 && (
        <div style={{ marginTop: 10 }}>
          <div style={{ fontSize: 9, color: 'var(--muted)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            GPU Processes
          </div>
          {gpu.processes.map((proc) => (
            <GpuProcessRow key={proc.pid} proc={proc} onKill={onKillProcess} />
          ))}
        </div>
      )}
    </MonPanel>
  )
}

function PlaceholderGpuPanel({ index, color }: { index: number; color: string }) {
  return (
    <MonPanel title={`GPU ${index}`} titleColor={color} accentColor={color}>
      <div style={{ fontSize: 11, color: 'var(--muted)', padding: '16px 0', textAlign: 'center' }}>
        等待数据...
      </div>
    </MonPanel>
  )
}

function MonPanel({
  title,
  titleColor,
  value,
  accentColor,
  children,
}: {
  title: string
  titleColor?: string
  value?: string
  accentColor?: string
  children: React.ReactNode
}) {
  return (
    <div
      className="rounded-md"
      style={{
        background: 'var(--card)',
        border: '1px solid var(--border)',
        borderTop: accentColor ? `2px solid ${accentColor}` : undefined,
        padding: 16,
      }}
    >
      <div className="flex items-center gap-2 mb-2" style={{ fontSize: 12, fontWeight: 600, color: titleColor ?? 'var(--accent-2)' }}>
        {title}
        {value && (
          <span className="ml-auto" style={{ color: 'var(--text-strong)', fontFamily: 'var(--mono)', fontSize: 14 }}>
            {value}
          </span>
        )}
      </div>
      {children}
    </div>
  )
}

function GpuProcessRow({ proc, onKill }: { proc: GpuProcessInfo; onKill: (pid: number, memMb: number) => void }) {
  const memG = (proc.used_gpu_memory_mb / 1024).toFixed(1)
  const isOrphan = !proc.managed

  return (
    <div
      className="flex items-center gap-2"
      style={{
        padding: '3px 6px',
        fontSize: 11,
        fontFamily: 'var(--mono)',
        color: isOrphan ? 'var(--warn)' : 'var(--muted)',
        borderBottom: '1px solid var(--border)',
      }}
    >
      <span style={{ width: 60, flexShrink: 0 }}>{proc.pid}</span>
      <span style={{ width: 50, flexShrink: 0 }}>{memG}G</span>
      <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {proc.managed ? proc.model_name : proc.command}
      </span>
      {proc.managed ? (
        <span style={{
          fontSize: 9,
          padding: '1px 5px',
          borderRadius: 3,
          background: 'color-mix(in srgb, var(--ok) 15%, transparent)',
          color: 'var(--ok)',
          flexShrink: 0,
        }}>
          managed
        </span>
      ) : (
        <>
          <span style={{
            fontSize: 9,
            padding: '1px 5px',
            borderRadius: 3,
            background: 'color-mix(in srgb, var(--warn) 15%, transparent)',
            color: 'var(--warn)',
            flexShrink: 0,
          }}>
            orphan
          </span>
          <button
            onClick={() => onKill(proc.pid, proc.used_gpu_memory_mb)}
            title={`Kill process ${proc.pid}`}
            style={{
              background: 'color-mix(in srgb, var(--accent) 15%, transparent)',
              border: '1px solid color-mix(in srgb, var(--accent) 30%, transparent)',
              borderRadius: 3,
              padding: '1px 4px',
              cursor: 'pointer',
              color: 'var(--accent)',
              display: 'flex',
              alignItems: 'center',
              flexShrink: 0,
            }}
          >
            <X size={10} />
          </button>
        </>
      )}
    </div>
  )
}

function ProcRow({ pid, cpu, mem, name, cmd, index }: { pid: string; cpu: string; mem: string; name: string; cmd: string; index: number }) {
  return (
    <div
      className="grid"
      style={{
        gridTemplateColumns: '60px 60px 60px 80px 1fr',
        gap: 4,
        fontSize: 12,
        color: 'var(--muted)',
        fontFamily: 'var(--mono)',
        padding: '3px 0',
        background: index % 2 === 1 ? 'rgba(255,255,255,0.02)' : 'transparent',
        borderBottom: '1px solid rgba(255,255,255,0.02)',
      }}
    >
      <span>{pid}</span>
      <span>{cpu}</span>
      <span style={{ color: 'var(--warn)' }}>{mem}</span>
      <span>{name}</span>
      <span style={{ color: 'var(--ok)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{cmd}</span>
    </div>
  )
}
