import { useEffect, useRef, useState } from 'react'
import { Ban, Check, LoaderCircle, XCircle } from 'lucide-react'
import { cancelPrediction, getPrediction, type Prediction } from '../../api/predictions'

export interface AsyncRunStateProps {
  predictionId: string
  /** succeeded 终态触发一次,带上 prediction.output(供调用方直接喂给 SchemaDrivenOutput)。 */
  onDone: (output: Record<string, unknown> | null) => void
  /** 用户点「取消」且 cancelPrediction 请求已结束(无论成败)后触发一次。 */
  onCancel: () => void
}

const POLL_MS = 2000

const TERMINAL: ReadonlySet<Prediction['status']> = new Set(['succeeded', 'failed', 'canceled'])

// 四段进度条(spec 2026-08-10 task-10):已提交(mount 即达成)→ 排队中(starting)→
// 运行中(processing,带耗时秒表)→ 完成(终态,succeeded/failed/canceled 三种落地文案)。
const STAGE_LABELS = ['已提交', '排队中', '运行中', '完成'] as const

function stageIndex(status: Prediction['status'] | undefined): number {
  switch (status) {
    case 'starting': return 1
    case 'processing': return 2
    case 'succeeded':
    case 'failed':
    case 'canceled':
      return 3
    default:
      return 0
  }
}

function tickElapsed(sinceIso: string | null, now: number): string {
  if (!sinceIso) return '0:00'
  const start = new Date(sinceIso).getTime()
  const sec = Math.max(0, Math.floor((now - start) / 1000))
  const m = Math.floor(sec / 60)
  const s = sec % 60
  return `${m}:${String(s).padStart(2, '0')}`
}

/** Playground 异步运行态(comfy_template respond-async 服务):2s 轮询 prediction 状态,
 * 渲染分段进度 + 取消按钮;终态 succeeded 把 output 交给调用方渲染,failed 展示 error 文案。 */
export default function AsyncRunState({ predictionId, onDone, onCancel }: AsyncRunStateProps) {
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [pollError, setPollError] = useState<string | null>(null)
  const [cancelling, setCancelling] = useState(false)
  const [now, setNow] = useState(() => Date.now())
  const doneCalledRef = useRef(false)

  useEffect(() => {
    let alive = true
    let timer: ReturnType<typeof setInterval> | null = null

    const poll = async () => {
      try {
        const pred = await getPrediction(predictionId)
        if (!alive) return
        setPollError(null)
        setPrediction(pred)
        if (TERMINAL.has(pred.status)) {
          if (timer) clearInterval(timer)
          if (pred.status === 'succeeded' && !doneCalledRef.current) {
            doneCalledRef.current = true
            onDone(pred.output)
          }
        }
      } catch (e) {
        // 轮询偶发网络抖动:留住上一帧,不中断轮询,下一轮重试。
        if (alive) setPollError((e as Error).message ?? String(e))
      }
    }

    poll()
    timer = setInterval(poll, POLL_MS)
    return () => {
      alive = false
      if (timer) clearInterval(timer)
    }
    // predictionId 变化才重开一轮轮询;onDone 由调用方每次渲染新建也不该重触发。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [predictionId])

  // 运行中耗时秒表:仅在未到终态时每秒 tick。
  useEffect(() => {
    if (!prediction || TERMINAL.has(prediction.status)) return
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [prediction?.status])

  const handleCancel = async () => {
    setCancelling(true)
    try {
      await cancelPrediction(predictionId)
    } catch (e) {
      setPollError((e as Error).message ?? String(e))
    } finally {
      setCancelling(false)
      onCancel()
    }
  }

  const status = prediction?.status
  const idx = stageIndex(status)
  const isTerminal = status !== undefined && TERMINAL.has(status)
  const progress = prediction?.progress

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, padding: '14px 18px' }}>
      <style>{`@keyframes asyncRunSpin{to{transform:rotate(360deg)}}`}</style>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {STAGE_LABELS.map((label, i) => {
          const passed = i < idx
          const current = i === idx
          const isFinalRow = i === 3
          let dotColor = 'var(--muted)'
          if (isFinalRow && current) {
            dotColor = status === 'failed' ? 'var(--error, #ef4444)'
              : status === 'canceled' ? 'var(--muted)'
              : 'var(--ok, #34c759)'
          } else if (passed || (current && !isFinalRow)) {
            dotColor = current ? 'var(--accent)' : 'var(--ok, #34c759)'
          }
          const rowLabel = isFinalRow && current
            ? (status === 'failed' ? '失败' : status === 'canceled' ? '已取消' : '完成')
            : label
          return (
            <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
              <span style={{
                width: 16, height: 16, borderRadius: '50%', flexShrink: 0,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                background: passed || (current && isFinalRow) ? dotColor : 'transparent',
                border: `1.5px solid ${dotColor}`,
                color: '#fff',
              }}>
                {passed ? (
                  <Check size={10} />
                ) : current && !isFinalRow ? (
                  <LoaderCircle size={10} style={{ animation: 'asyncRunSpin 1s linear infinite', color: dotColor }} />
                ) : current && isFinalRow && status === 'failed' ? (
                  <XCircle size={11} style={{ color: dotColor }} />
                ) : current && isFinalRow && status === 'canceled' ? (
                  <Ban size={10} style={{ color: dotColor }} />
                ) : current && isFinalRow ? (
                  <Check size={10} />
                ) : null}
              </span>
              <span style={{
                color: current ? 'var(--text)' : passed ? 'var(--text)' : 'var(--muted)',
                fontWeight: current ? 600 : 400,
              }}>
                {rowLabel}
              </span>
              {current && !isFinalRow && (
                <span style={{ color: 'var(--muted)', fontFamily: 'var(--mono, monospace)', fontSize: 11 }}>
                  {status === 'processing' ? tickElapsed(prediction?.created_at ?? null, now) : ''}
                  {status === 'processing' && progress && progress.nodes_total > 0
                    ? ` · ${progress.nodes_done}/${progress.nodes_total} 节点`
                    : ''}
                </span>
              )}
            </div>
          )
        })}
      </div>

      {status === 'failed' && prediction?.error && (
        <div style={{ color: 'var(--error, #ef4444)', fontSize: 12, lineHeight: 1.5 }}>
          {prediction.error}
        </div>
      )}
      {pollError && (
        <div style={{ color: 'var(--warn, #f59e0b)', fontSize: 11 }}>
          轮询异常(将自动重试):{pollError}
        </div>
      )}

      {!isTerminal && (
        <button
          type="button"
          onClick={handleCancel}
          disabled={cancelling}
          style={{
            alignSelf: 'flex-start', display: 'inline-flex', alignItems: 'center', gap: 6,
            padding: '5px 12px', borderRadius: 6, fontSize: 12,
            background: 'transparent', color: 'var(--error, #ef4444)',
            border: '1px solid var(--border)', cursor: cancelling ? 'not-allowed' : 'pointer',
            opacity: cancelling ? 0.6 : 1,
          }}
        >
          <Ban size={12} />
          {cancelling ? '取消中…' : '取消'}
        </button>
      )}
    </div>
  )
}
