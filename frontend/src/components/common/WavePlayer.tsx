import { useRef, useState, useEffect, useCallback } from 'react'
import { initWasm, decode_audio, compute_waveform } from '../../wasm'

interface Props {
  /** Base64-encoded audio data (WAV format) */
  audioBase64: string | null
  sampleRate?: number
  duration?: number
}

export default function WavePlayer({ audioBase64, sampleRate, duration }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [peaks, setPeaks] = useState<Float32Array | null>(null)
  const [totalDuration, setTotalDuration] = useState(duration ?? 0)
  const [wasmFailed, setWasmFailed] = useState(false)

  // Decode audio + compute waveform via WASM
  useEffect(() => {
    if (!audioBase64) { setPeaks(null); return }

    let cancelled = false
    ;(async () => {
      try {
        await initWasm()
        const raw = Uint8Array.from(atob(audioBase64), (c) => c.charCodeAt(0))
        const decoded = decode_audio(raw)
        if (cancelled) return
        const mono = decoded.samples_mono()
        setTotalDuration(decoded.duration_seconds)
        const p = compute_waveform(mono, 800)
        setPeaks(new Float32Array(p))
      } catch (e) {
        console.warn('WASM decode failed, falling back to audio element', e)
        setWasmFailed(true)
        // Fall back to HTML audio element for duration
        const audio = audioRef.current
        if (audio) {
          audio.onloadedmetadata = () => setTotalDuration(audio.duration)
        }
      }
    })()

    return () => { cancelled = true }
  }, [audioBase64])

  // Draw waveform
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !peaks) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const w = canvas.width / dpr
    const h = canvas.height / dpr
    const mid = h / 2
    const numBuckets = peaks.length / 2

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.scale(dpr, dpr)

    // Progress position
    const progress = totalDuration > 0 ? currentTime / totalDuration : 0
    const progressX = progress * w

    for (let i = 0; i < numBuckets; i++) {
      const x = (i / numBuckets) * w
      const min = peaks[i * 2]
      const max = peaks[i * 2 + 1]
      const barW = Math.max(1, w / numBuckets - 0.5)

      ctx.fillStyle = x < progressX ? '#3b82f6' : '#4b5563'
      ctx.fillRect(x, mid - max * mid, barW, (max - min) * mid || 1)
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0)
  }, [peaks, currentTime, totalDuration])

  // Resize canvas for HiDPI
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
  }, [peaks])

  // Time update
  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    const handler = () => setCurrentTime(audio.currentTime)
    audio.addEventListener('timeupdate', handler)
    audio.addEventListener('ended', () => setPlaying(false))
    return () => {
      audio.removeEventListener('timeupdate', handler)
      audio.removeEventListener('ended', () => setPlaying(false))
    }
  }, [])

  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    const audio = audioRef.current
    if (!canvas || !audio || !totalDuration) return
    const rect = canvas.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width
    audio.currentTime = x * totalDuration
    setCurrentTime(audio.currentTime)
  }, [totalDuration])

  const togglePlay = useCallback(() => {
    const audio = audioRef.current
    if (!audio) return
    if (playing) {
      audio.pause()
      setPlaying(false)
    } else {
      audio.play()
      setPlaying(true)
    }
  }, [playing])

  if (!audioBase64) return null

  const src = `data:audio/wav;base64,${audioBase64}`

  return (
    <div
      style={{
        background: 'var(--bg)',
        borderRadius: 6,
        padding: 8,
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
      }}
    >
      <audio ref={audioRef} src={src} preload="metadata" />

      {/* Waveform canvas */}
      {peaks ? (
        <canvas
          ref={canvasRef}
          style={{
            width: '100%',
            height: 48,
            cursor: 'pointer',
            borderRadius: 4,
            background: 'var(--card)',
          }}
          onClick={handleCanvasClick}
        />
      ) : wasmFailed ? (
        <div
          style={{
            width: '100%',
            height: 48,
            borderRadius: 4,
            background: 'var(--card)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 10,
            color: 'var(--muted)',
          }}
        >
          波形加载中...
        </div>
      ) : (
        <div
          style={{
            width: '100%',
            height: 48,
            borderRadius: 4,
            background: 'var(--card)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <div
            style={{
              width: 16,
              height: 16,
              border: '2px solid var(--accent)',
              borderTopColor: 'transparent',
              borderRadius: '50%',
              animation: 'spin 0.8s linear infinite',
            }}
          />
        </div>
      )}

      {/* Controls */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}
      >
        <button
          onClick={togglePlay}
          style={{
            padding: '3px 10px',
            fontSize: 10,
            borderRadius: 4,
            border: 'none',
            background: 'var(--accent)',
            color: '#fff',
            cursor: 'pointer',
            flexShrink: 0,
          }}
        >
          {playing ? '⏸' : '▶'}
        </button>

        <span style={{ fontSize: 10, color: 'var(--muted)', fontFamily: 'var(--mono)' }}>
          {currentTime.toFixed(1)}s / {totalDuration.toFixed(1)}s
        </span>

        {sampleRate && (
          <span style={{ fontSize: 9, color: 'var(--muted-strong)' }}>
            {sampleRate}Hz
          </span>
        )}

        <a
          href={src}
          download="tts_output.wav"
          style={{
            fontSize: 10,
            color: 'var(--accent)',
            textDecoration: 'none',
            marginLeft: 'auto',
          }}
        >
          下载
        </a>
      </div>
    </div>
  )
}
