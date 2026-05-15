import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { useNotificationStore } from './notifications'

describe('notifications store', () => {
  beforeEach(() => {
    useNotificationStore.setState({ notified: new Set(), permission: 'default' })
    vi.restoreAllMocks()
  })
  afterEach(() => vi.restoreAllMocks())

  it('notifyOnce dedupes by task_id', () => {
    const toast = vi.fn()
    const { notifyOnce } = useNotificationStore.getState()
    notifyOnce('task-1', 'flux2 完成 · 34s', 'success', toast)
    notifyOnce('task-1', 'flux2 完成 · 34s', 'success', toast)
    expect(toast).toHaveBeenCalledTimes(1)
  })

  it('always fires the in-app toast even without notification permission', () => {
    const toast = vi.fn()
    useNotificationStore.setState({ permission: 'denied' })
    useNotificationStore.getState().notifyOnce('task-2', '失败', 'error', toast)
    expect(toast).toHaveBeenCalledWith('失败', 'error')
  })

  it('only sends a system Notification when the page is unfocused + permission granted', () => {
    const NotificationCtor = vi.fn()
    vi.stubGlobal('Notification', Object.assign(NotificationCtor, { permission: 'granted' }))
    vi.spyOn(document, 'hasFocus').mockReturnValue(false) // 页面失焦
    useNotificationStore.setState({ permission: 'granted' })

    useNotificationStore.getState().notifyOnce('task-3', '完成', 'success', vi.fn())
    expect(NotificationCtor).toHaveBeenCalledTimes(1)

    // 页面有焦点时不发系统通知（只 toast）
    vi.spyOn(document, 'hasFocus').mockReturnValue(true)
    useNotificationStore.getState().notifyOnce('task-4', '完成', 'success', vi.fn())
    expect(NotificationCtor).toHaveBeenCalledTimes(1) // 没再增加
  })

  it('requestPermission updates store + degrades gracefully when API absent', async () => {
    // 无 Notification API 的环境（jsdom 默认）→ permission 落 'denied'
    vi.stubGlobal('Notification', undefined)
    await useNotificationStore.getState().requestPermission()
    expect(useNotificationStore.getState().permission).toBe('denied')
  })
})
