import { describe, it, expect, vi } from 'vitest'
import {
  NOUS_IMAGE_URL_MIME,
  setImageDrag,
  readImageDropUrl,
  isDisplayableImageValue,
} from './imageDragDrop'

// 极简 DataTransfer mock(jsdom 的 DataTransfer 不存数据)。
function makeDataTransfer(initial: Record<string, string> = {}) {
  const store: Record<string, string> = { ...initial }
  return {
    effectAllowed: '',
    setData: (type: string, val: string) => { store[type] = val },
    getData: (type: string) => store[type] ?? '',
  } as unknown as DataTransfer
}

describe('imageDragDrop —— 出图拖到输入契约', () => {
  describe('setImageDrag', () => {
    it('双写自定义 MIME + text/uri-list,并停止冒泡(避免 React Flow 拖节点)', () => {
      const dt = makeDataTransfer()
      const stopPropagation = vi.fn()
      const e = { dataTransfer: dt, stopPropagation } as unknown as React.DragEvent
      setImageDrag(e, '/files/images/2026-06-12/abc.png?token=x')
      expect(dt.getData(NOUS_IMAGE_URL_MIME)).toBe('/files/images/2026-06-12/abc.png?token=x')
      expect(dt.getData('text/uri-list')).toBe('/files/images/2026-06-12/abc.png?token=x')
      expect(dt.effectAllowed).toBe('copy')
      expect(stopPropagation).toHaveBeenCalled()
    })
  })

  describe('readImageDropUrl', () => {
    it('优先读自定义 MIME', () => {
      const dt = makeDataTransfer({
        [NOUS_IMAGE_URL_MIME]: '/files/images/a.png',
        'text/uri-list': 'https://other/b.png',
      })
      expect(readImageDropUrl(dt)).toBe('/files/images/a.png')
    })

    it('无自定义 MIME 时回退 text/uri-list', () => {
      const dt = makeDataTransfer({ 'text/uri-list': 'https://x/c.png' })
      expect(readImageDropUrl(dt)).toBe('https://x/c.png')
    })

    it('text/uri-list 跳过 # 注释行,取首条 URL', () => {
      const dt = makeDataTransfer({ 'text/uri-list': '# comment\nhttps://x/d.png\nhttps://x/e.png' })
      expect(readImageDropUrl(dt)).toBe('https://x/d.png')
    })

    it('无 URL(拖的是本地文件)→ null,调用方回退 File 读取', () => {
      expect(readImageDropUrl(makeDataTransfer())).toBeNull()
    })
  })

  describe('isDisplayableImageValue', () => {
    it.each([
      ['data:image/png;base64,iVBOR', true],
      ['/files/images/2026/a.png?token=x', true],
      ['https://cdn/x.png', true],
      ['http://h/x.png', true],
      ['', false],
      [undefined, false],
      [null, false],
      ['just-a-name', false],
    ])('%s → %s', (val, expected) => {
      expect(isDisplayableImageValue(val as string | undefined | null)).toBe(expected)
    })
  })
})
