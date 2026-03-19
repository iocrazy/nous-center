import { useSysGpus, useSysStats, useSysProcesses, useMonitorStats, type SysGpuInfo } from '../../api/system'

export default function DashboardOverlay() {
  const { data: gpuData } = useSysGpus()
  const { data: sysStats } = useSysStats()
  const { data: procData } = useSysProcesses()
  const { data: monitorData } = useMonitorStats()

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
      <div style={{ padding: 10 }}>
        {/* Stats row */}
        <div className="grid grid-cols-4 gap-2 mb-2">
          <StatCard label="API Keys" value="3" sub="2 active" />
          <StatCard label="Today Calls" value="--" sub="--" />
          <StatCard label="Uptime" value={monitorData ? formatUptime(monitorData.uptime_seconds) : '--'} sub="" />
          <StatCard label="Token Usage" value="--" sub="--" />
        </div>

        {/* GPU panels */}
        <div className="grid grid-cols-2 gap-2 mb-2">
          {gpuData?.gpus ? (
            gpuData.gpus.map((gpu, i) => (
              <GpuPanel
                key={gpu.index}
                gpu={gpu}
                chartColor={i === 0 ? 'var(--ok)' : 'var(--accent-2)'}
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
        <div className="grid grid-cols-4 gap-2 mb-2">
          <StatCard
            label="CPU"
            value={sysStats ? `${fmt(sysStats.cpu_usage_percent, 0)}%` : '--'}
            sub={sysStats ? `${sysStats.cpu_count} cores` : '--'}
          />
          <StatCard
            label="RAM"
            value={sysStats ? `${fmt(sysStats.memory_used_gb)}G` : '--'}
            sub={sysStats ? `/ ${fmt(sysStats.memory_total_gb)}G` : '--'}
          />
          <StatCard
            label="SWAP"
            value={sysStats ? `${fmt(sysStats.swap_used_gb)}G` : '--'}
            sub={sysStats ? `/ ${fmt(sysStats.swap_total_gb)}G` : '--'}
          />
          <StatCard
            label="Disk"
            value={sysStats ? `${fmt(sysStats.disk_used_gb)}G` : '--'}
            sub={sysStats ? `/ ${fmt(sysStats.disk_total_gb)}G` : '--'}
          />
        </div>

        {/* Process table */}
        <MonPanel title="Processes">
          <div
            className="grid"
            style={{
              gridTemplateColumns: '50px 50px 50px 50px 1fr',
              gap: 4,
              fontSize: 8,
              color: 'var(--accent-2)',
              fontFamily: 'var(--mono)',
              padding: '3px 0',
              borderBottom: '1px solid var(--border)',
              fontWeight: 600,
            }}
          >
            <span>PID</span><span>CPU%</span><span>MEM</span><span>NAME</span><span>COMMAND</span>
          </div>
          {procData?.processes ? (
            procData.processes.slice(0, 15).map((p) => (
              <ProcRow
                key={p.pid}
                pid={String(p.pid)}
                cpu={`${p.cpu_percent.toFixed(1)}%`}
                mem={`${p.memory_mb}M`}
                name={p.name}
                cmd={p.command}
              />
            ))
          ) : (
            <div style={{ fontSize: 9, color: 'var(--muted)', padding: '8px 0', textAlign: 'center' }}>
              启动 nous-center-sys 以获取进程数据
            </div>
          )}
        </MonPanel>
      </div>
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
        padding: '8px 10px',
      }}
    >
      <div style={{ fontSize: 8, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 3 }}>
        {label}
      </div>
      <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-strong)', fontFamily: 'var(--mono)' }}>
        {value}
      </div>
      <div style={{ fontSize: 8, color: 'var(--ok)', marginTop: 1 }}>{sub}</div>
    </div>
  )
}

function GpuPanel({ gpu, chartColor }: { gpu: SysGpuInfo; chartColor: string }) {
  const memPct = gpu.memory_total_mb > 0 ? (gpu.memory_used_mb / gpu.memory_total_mb) * 100 : 0
  const memUsedG = (gpu.memory_used_mb / 1024).toFixed(1)
  const memTotalG = (gpu.memory_total_mb / 1024).toFixed(0)

  return (
    <MonPanel title={`GPU ${gpu.index}`} titleColor={chartColor} value={`${gpu.utilization_gpu}%`}>
      <div style={{ fontSize: 8, color: 'var(--muted-strong)', fontFamily: 'var(--mono)', lineHeight: 1.6, marginBottom: 4 }}>
        <span style={{ color: 'var(--text)' }}>{gpu.name}</span>
        <br />
        <span style={{ color: 'var(--accent-2)' }}>TEMP</span>{' '}
        <span style={{ color: gpu.temperature < 50 ? 'var(--ok)' : gpu.temperature < 80 ? 'var(--warn)' : '#ef4444' }}>
          {gpu.temperature}°C
        </span> &nbsp;
        <span style={{ color: 'var(--accent-2)' }}>FAN</span> {gpu.fan_speed}% &nbsp;
        <span style={{ color: 'var(--accent-2)' }}>POW</span> {gpu.power_draw_w.toFixed(0)}/{gpu.power_limit_w.toFixed(0)}W
      </div>

      {/* Simple utilization bar */}
      <div className="flex items-center gap-1.5 mb-1">
        <span style={{ fontSize: 9, color: 'var(--muted)', width: 32, fontFamily: 'var(--mono)' }}>GPU</span>
        <div className="flex-1 overflow-hidden" style={{ height: 10, background: 'var(--bg)', borderRadius: 2 }}>
          <div
            style={{
              height: '100%',
              width: `${gpu.utilization_gpu}%`,
              background: chartColor,
              borderRadius: 2,
              transition: 'width 0.5s',
            }}
          />
        </div>
        <span style={{ fontSize: 8, color: 'var(--muted)', fontFamily: 'var(--mono)', width: 32, textAlign: 'right' }}>
          {gpu.utilization_gpu}%
        </span>
      </div>

      {/* Memory bar */}
      <div className="flex items-center gap-1.5 mb-0.5">
        <span style={{ fontSize: 9, color: 'var(--muted)', width: 32, fontFamily: 'var(--mono)' }}>MEM</span>
        <div className="flex-1 overflow-hidden" style={{ height: 10, background: 'var(--bg)', borderRadius: 2 }}>
          <div
            style={{
              height: '100%',
              width: `${memPct}%`,
              background: chartColor,
              borderRadius: 2,
              transition: 'width 0.5s',
            }}
          />
        </div>
        <span style={{ fontSize: 8, color: 'var(--muted)', fontFamily: 'var(--mono)', width: 56, textAlign: 'right' }}>
          {memUsedG}G / {memTotalG}G
        </span>
      </div>
    </MonPanel>
  )
}

function PlaceholderGpuPanel({ index, color }: { index: number; color: string }) {
  return (
    <MonPanel title={`GPU ${index}`} titleColor={color}>
      <div style={{ fontSize: 9, color: 'var(--muted)', padding: '12px 0', textAlign: 'center' }}>
        等待数据...
      </div>
    </MonPanel>
  )
}

function MonPanel({
  title,
  titleColor,
  value,
  children,
}: {
  title: string
  titleColor?: string
  value?: string
  children: React.ReactNode
}) {
  return (
    <div
      className="rounded-md"
      style={{
        background: 'var(--card)',
        border: '1px solid var(--border)',
        padding: '8px 10px',
      }}
    >
      <div className="flex items-center gap-1.5 mb-1.5" style={{ fontSize: 10, fontWeight: 600, color: titleColor ?? 'var(--accent-2)' }}>
        {title}
        {value && (
          <span className="ml-auto" style={{ color: 'var(--text-strong)', fontFamily: 'var(--mono)', fontSize: 11 }}>
            {value}
          </span>
        )}
      </div>
      {children}
    </div>
  )
}

function ProcRow({ pid, cpu, mem, name, cmd }: { pid: string; cpu: string; mem: string; name: string; cmd: string }) {
  return (
    <div
      className="grid"
      style={{
        gridTemplateColumns: '50px 50px 50px 50px 1fr',
        gap: 4,
        fontSize: 8,
        color: 'var(--muted)',
        fontFamily: 'var(--mono)',
        padding: '2px 0',
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
