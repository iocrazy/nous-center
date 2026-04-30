import { useEffect, useRef } from 'react'

export interface LiveChannelOptions {
  /** Called for every JSON-parsed message. Non-JSON frames are dropped. */
  onMessage?: (msg: Record<string, unknown>) => void
  /**
   * Called every time the socket re-opens after the first connection.
   * Use this to trigger a fresh data fetch — events that fired during
   * the disconnect are missed by definition, so re-reading the
   * authoritative state is the only way to get back in sync.
   */
  onReconnect?: () => void
}

const RECONNECT_BACKOFF_MS = [1_000, 2_000, 5_000, 15_000] as const

/**
 * Per-URL shared connection. Multiple components calling
 * `useLiveChannel('/ws/models', ...)` should NOT spawn separate sockets —
 * they all want the same broadcast stream. The shared connection also
 * lets sub-trees that don't directly mount the owning hook (e.g. canvas
 * dropdowns vs. the Models overlay) ride on the same updates.
 */
interface SharedChannel {
  socket: WebSocket | null
  consumers: Set<{
    onMessage?: (msg: Record<string, unknown>) => void
    onReconnect?: () => void
  }>
  attempt: number
  reconnectTimer: number | null
  hasOpenedOnce: boolean
}

const channels = new Map<string, SharedChannel>()

function getOrCreate(url: string): SharedChannel {
  const existing = channels.get(url)
  if (existing) return existing
  const fresh: SharedChannel = {
    socket: null,
    consumers: new Set(),
    attempt: 0,
    reconnectTimer: null,
    hasOpenedOnce: false,
  }
  channels.set(url, fresh)
  return fresh
}

function clearTimer(ch: SharedChannel) {
  if (ch.reconnectTimer !== null) {
    window.clearTimeout(ch.reconnectTimer)
    ch.reconnectTimer = null
  }
}

function connect(url: string) {
  const ch = channels.get(url)
  if (!ch) return
  if (ch.consumers.size === 0) return // no one's listening

  const ws = new WebSocket(url)
  ch.socket = ws

  ws.onopen = () => {
    ch.attempt = 0
    if (ch.hasOpenedOnce) {
      for (const c of ch.consumers) c.onReconnect?.()
    }
    ch.hasOpenedOnce = true
  }

  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data)
      for (const c of ch.consumers) c.onMessage?.(msg)
    } catch {
      // Non-JSON frames are not part of our protocol; drop silently.
    }
  }

  ws.onerror = () => {
    // onclose fires next; reconnect bookkeeping happens there.
  }

  ws.onclose = () => {
    ch.socket = null
    if (ch.consumers.size === 0) return // no one cares anymore
    const delay = RECONNECT_BACKOFF_MS[
      Math.min(ch.attempt, RECONNECT_BACKOFF_MS.length - 1)
    ]
    ch.attempt += 1
    clearTimer(ch)
    ch.reconnectTimer = window.setTimeout(() => connect(url), delay)
  }
}

function ensureVisibilityHook() {
  // One global listener; multiplexes across all channels.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  if ((window as any).__liveChannelVisHook) return
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).__liveChannelVisHook = true
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState !== 'visible') return
    // Tab just came back to foreground — kick any disconnected channels.
    for (const [url, ch] of channels) {
      if (ch.consumers.size === 0) continue
      const s = ch.socket
      if (s && (s.readyState === WebSocket.OPEN || s.readyState === WebSocket.CONNECTING)) {
        continue
      }
      clearTimer(ch)
      ch.attempt = 0
      connect(url)
    }
  })
}

/**
 * Long-lived WebSocket channel with exponential-backoff reconnect and
 * tab-visibility awareness. Use for global push streams that need to
 * stay connected for the whole session — e.g. /ws/models, /ws/tasks.
 *
 * Multiple components subscribing to the same URL share one socket; the
 * connection is released only when the last consumer unmounts. This
 * keeps canvas dropdowns and the Models overlay both live-updating
 * without doubling network connections.
 *
 * Not for short-lived per-run/per-task channels. Those have their own
 * lifecycle bound to the work item; reconnecting to a session that has
 * already ended just churns.
 */
export function useLiveChannel(
  url: string,
  { onMessage, onReconnect }: LiveChannelOptions = {},
): void {
  // Stash callbacks in refs so changing them doesn't tear down the socket.
  const onMessageRef = useRef(onMessage)
  const onReconnectRef = useRef(onReconnect)
  onMessageRef.current = onMessage
  onReconnectRef.current = onReconnect

  useEffect(() => {
    ensureVisibilityHook()

    const ch = getOrCreate(url)
    const consumer = {
      onMessage: (m: Record<string, unknown>) => onMessageRef.current?.(m),
      onReconnect: () => onReconnectRef.current?.(),
    }
    ch.consumers.add(consumer)

    if (!ch.socket) {
      clearTimer(ch)
      connect(url)
    }

    return () => {
      ch.consumers.delete(consumer)
      if (ch.consumers.size === 0) {
        clearTimer(ch)
        // 1000 = normal closure → server side won't log it as an error,
        // and we won't try to reconnect on this empty channel.
        ch.socket?.close(1000)
        ch.socket = null
        channels.delete(url)
      }
    }
  }, [url])
}
