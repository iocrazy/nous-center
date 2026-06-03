/**
 * TopbarGpuMonitor — 顶栏紧凑硬件监控条(对齐 ComfyUI 风格,跟 infra healthy 同层)。
 *
 * 每张卡:短名 + util 进度条 + util% + 温度。点击整条展开/收起显存(used/total)。
 * 数据走 useGpuStats(/api/v1/monitor/stats,2s 轮询,nvidia-smi)。GPU-only —— 后端
 * monitor 暂不采 CPU/内存(要加需 psutil)。多卡场景一次显示全部卡(用户库:Pro6000 + 2×3090)。
 */
import { useState } from 'react'
import { useGpuStats, useSystemStats, type GpuInfo } from '../../api/gpuStats'

function shortName(name: string): string {
  const n = (name || '').replace(/NVIDIA\s+/i, '').replace(/GeForce\s+/i, '')
  if (/pro\s*6000/i.test(n)) return 'Pro6000'
  const m = n.match(/RTX\s*(\d{3,4}\w*)/i)
  if (m) return m[1] // "3090"
  return n.slice(0, 8) || `GPU${name}`
}

/** 重名(两张 3090)用 ·1 ·2 区分,按 nvidia-smi index 顺序。 */
function labelFor(gpus: GpuInfo[], g: GpuInfo): string {
  const base = shortName(g.name)
  const dupes = gpus.filter((x) => shortName(x.name) === base)
  if (dupes.length > 1) {
    const pos = dupes.findIndex((x) => x.index === g.index) + 1
    return `${base}·${pos}`
  }
  return base
}

function loadColor(pct: number): string {
  if (pct >= 85) return '#f87171'
  if (pct >= 60) return '#fbbf24'
  return 'var(--status-running, #4ade80)'
}
function tempColor(t: number): string {
  if (t >= 80) return '#f87171'
  if (t >= 65) return '#fbbf24'
  return 'var(--tp-text-muted)'
}

/** CPU / RAM 紧凑芯片:标签 + mini 进度条 + 百分比(+ 展开时附带详情)。 */
function MetricChip({ label, pct, detail }: { label: string; pct: number; detail?: string }) {
  const p = Math.max(0, Math.min(100, Math.round(pct)))
  return (
    <div className="flex items-center gap-1.5 text-[11px] leading-none" style={{ color: 'var(--tp-text-muted)' }}>
      <span style={{ fontWeight: 500, color: 'var(--tp-text)' }}>{label}</span>
      <span
        style={{
          width: 26, height: 5, borderRadius: 3,
          background: 'var(--tp-border-faint)', overflow: 'hidden',
          display: 'inline-block', flexShrink: 0,
        }}
      >
        <span style={{ display: 'block', height: '100%', width: `${p}%`, background: loadColor(p), transition: 'width .4s ease' }} />
      </span>
      <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: 28, textAlign: 'right' }}>{p}%</span>
      {detail && <span style={{ fontVariantNumeric: 'tabular-nums', opacity: 0.85 }}>{detail}</span>}
    </div>
  )
}

export default function TopbarGpuMonitor() {
  const { data: gpus } = useGpuStats()
  const { data: sys } = useSystemStats()
  const [expanded, setExpanded] = useState(false)
  if ((!gpus || gpus.length === 0) && !sys) return null

  return (
    <div
      className="flex items-center gap-3 mr-3 cursor-pointer select-none"
      onClick={() => setExpanded((e) => !e)}
      title={expanded ? '点击收起详情' : '点击展开显存/内存'}
      role="button"
      aria-label="硬件监控(CPU/内存/GPU)"
    >
      {sys && (
        <>
          <MetricChip label="CPU" pct={sys.cpu_usage_percent} />
          <MetricChip
            label="内存"
            pct={sys.memory_total_gb ? (sys.memory_used_gb / sys.memory_total_gb) * 100 : 0}
            detail={expanded ? `${sys.memory_used_gb.toFixed(0)}/${sys.memory_total_gb.toFixed(0)}G` : undefined}
          />
          {gpus && gpus.length > 0 && (
            <span style={{ width: 1, height: 14, background: 'var(--tp-border-faint)' }} />
          )}
        </>
      )}
      {(gpus ?? []).map((g, _i, arr) => {
        const util = Math.max(0, Math.min(100, g.utilization_gpu ?? 0))
        return (
          <div
            key={g.index}
            className="flex items-center gap-1.5 text-[11px] leading-none"
            style={{ color: 'var(--tp-text-muted)' }}
          >
            <span style={{ fontWeight: 500, color: 'var(--tp-text)' }}>{labelFor(arr, g)}</span>
            <span
              style={{
                width: 26, height: 5, borderRadius: 3,
                background: 'var(--tp-border-faint)', overflow: 'hidden',
                display: 'inline-block', flexShrink: 0,
              }}
            >
              <span
                style={{
                  display: 'block', height: '100%', width: `${util}%`,
                  background: loadColor(util), transition: 'width .4s ease',
                }}
              />
            </span>
            <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: 28, textAlign: 'right' }}>
              {util}%
            </span>
            <span style={{ color: tempColor(g.temperature), fontVariantNumeric: 'tabular-nums' }}>
              {g.temperature}°
            </span>
            {expanded && g.memory_total_mb > 0 && (
              <span style={{ fontVariantNumeric: 'tabular-nums', opacity: 0.85 }}>
                {(g.memory_used_mb / 1024).toFixed(1)}/{(g.memory_total_mb / 1024).toFixed(0)}G
              </span>
            )}
          </div>
        )
      })}
    </div>
  )
}
