import { create } from 'zustand'
import type { SynthesizeResponse } from '../api/tts'

export interface HistoryEntry {
  id: string
  text: string
  engine: string
  response: SynthesizeResponse
  timestamp: number
}

interface HistoryState {
  entries: HistoryEntry[]
  add: (entry: HistoryEntry) => void
  clear: () => void
}

export const useHistoryStore = create<HistoryState>((set) => ({
  entries: [],
  add: (entry) => set((s) => ({ entries: [entry, ...s.entries].slice(0, 50) })),
  clear: () => set({ entries: [] }),
}))
