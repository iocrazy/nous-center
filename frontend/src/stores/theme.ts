import { create } from 'zustand'

type ThemeMode = 'dark' | 'light' | 'auto'

interface ThemeState {
  mode: ThemeMode
  setMode: (mode: ThemeMode) => void
}

function getSystemTheme(): 'dark' | 'light' {
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function applyTheme(mode: ThemeMode) {
  const resolved = mode === 'auto' ? getSystemTheme() : mode
  document.documentElement.setAttribute('data-theme', resolved)
  localStorage.setItem('nous-theme', mode)
}

const stored = (localStorage.getItem('nous-theme') as ThemeMode) ?? 'dark'
applyTheme(stored)

export const useThemeStore = create<ThemeState>((set) => ({
  mode: stored,
  setMode: (mode) => {
    applyTheme(mode)
    set({ mode })
  },
}))

// Listen for system theme changes when in auto mode
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  const { mode } = useThemeStore.getState()
  if (mode === 'auto') applyTheme('auto')
})
