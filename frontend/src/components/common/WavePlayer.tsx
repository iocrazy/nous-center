import { useRef, useState, useEffect, useCallback } from 'react'
import { initWasm, decode_audio, compute_waveform } from '../../wasm'

interface Props {
  /** Base64-encoded audio data */
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

  // Decode audio + compute waveform via WASM
  useEffect(() => {
    if (!audioBase64) { setPeaks(null); return }

    let cancelled = false
    ;(async () => {
      await initWasm()
      const raw = Uint8Array.from(atob(audioBase64), (c) => c.charCodeAt(0))
      const decoded = decode_audio(raw)
      if (cancelled) return
      const mono = decoded.samples_mono()
      setTotalDuration(decoded.duration_seconds)
      const p = compute_waveform(mono, 800)
      setPeaks(new Float32Array(p))
    })()

    return () => { cancelled = true }
  }, [audioBase64])

  // Draw waveform
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !peaks) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const w = canvas.width
    const h = canvas.height
    const mid = h / 2
    const numBuckets = peaks.length / 2

    ctx.clearRect(0, 0, w, h)

    // Progress position
    const progress = totalDuration > 0 ? currentTime / totalDuration : 0
    const progressX = progress * w

    for (let i = 0; i < numBuckets; i++) {
      const x = (i / numBuckets) * w
      const min = peaks[i * 2]
      const max = peaks[i * 2 + 1]

      ctx.fillStyle = x < progressX ? '#3b82f6' : '#4b5563'
      ctx.fillRect(x, mid - max * mid, Math.max(1, w / numBuckets), (max - min) * mid)
    }
  }, [peaks, currentTime, totalDuration])

  // Time update
  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    const handler = () => setCurrentTime(audio.currentTime)
    audio.addEventListener('timeupdate', handler)
    return () => audio.removeEventListener('timeupdate', handler)
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

  if (!audioBase64) return null

  const src = `data:audio/wav;base64,${audioBase64}`

  return (
    <div className="bg-gray-900 rounded p-3 space-y-2">
      <audio
        ref={audioRef}
        src={src}
        onEnded={() => setPlaying(false)}
      />
      <canvas
        ref={canvasRef}
        width={800}
        height={80}
        className="w-full h-20 cursor-pointer rounded"
        onClick={handleCanvasClick}
      />
      <div className="flex items-center gap-3">
        <button
          className="px-3 py-1 bg-blue-600 hover:bg-blue-500 rounded text-sm"
          onClick={() => {
            const audio = audioRef.current
            if (!audio) return
            if (playing) { audio.pause(); setPlaying(false) }
            else { audio.play(); setPlaying(true) }
          }}
        >
          {playing ? '⏸ 暂停' : '▶ 播放'}
        </button>
        <span className="text-xs text-gray-400">
          {currentTime.toFixed(1)}s / {totalDuration.toFixed(1)}s
        </span>
        {sampleRate && (
          <span className="text-xs text-gray-500">{sampleRate}Hz</span>
        )}
        <a
          href={src}
          download="tts_output.wav"
          className="text-xs text-blue-400 hover:text-blue-300 ml-auto"
        >
          下载
        </a>
      </div>
    </div>
  )
}
