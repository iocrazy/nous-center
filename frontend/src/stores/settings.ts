import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface SettingsState {
  localModelsPath: string
  cosyvoiceRepoPath: string
  indexttsRepoPath: string
  redisUrl: string
  apiBaseUrl: string

  update: (values: Partial<Omit<SettingsState, 'update' | 'reset'>>) => void
  reset: () => void
}

const DEFAULTS = {
  localModelsPath: '/media/heygo/program/models',
  cosyvoiceRepoPath: '/media/heygo/program/projects-code/github-repos/CosyVoice',
  indexttsRepoPath: '/media/heygo/program/projects-code/github-repos/index-tts',
  redisUrl: 'redis://localhost:6379/0',
  apiBaseUrl: 'http://localhost:8000',
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      ...DEFAULTS,
      update: (values) => set(values),
      reset: () => set(DEFAULTS),
    }),
    { name: 'nous-settings' },
  ),
)
