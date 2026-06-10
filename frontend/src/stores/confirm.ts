import { create } from 'zustand'

// 统一确认弹窗(替换原生 window.confirm 的丑框)。命令式用法:
//   if (await confirmDialog({ message: '确认删除?', danger: true })) { ... }
// 单例 host(ConfirmHost)挂在 App 根,渲染当前请求。

export interface ConfirmOptions {
  title?: string
  message: string
  confirmText?: string
  cancelText?: string
  /** 危险操作:确认按钮红色(删除/卸载/下线等不可撤销操作)。 */
  danger?: boolean
}

interface PendingConfirm extends ConfirmOptions {
  resolve: (ok: boolean) => void
}

interface ConfirmState {
  current: PendingConfirm | null
  open: (opts: ConfirmOptions) => Promise<boolean>
  resolve: (ok: boolean) => void
}

export const useConfirmStore = create<ConfirmState>((set, get) => ({
  current: null,
  open: (opts) =>
    new Promise<boolean>((resolve) => {
      // 若已有一个在等,先取消它(返回 false),避免请求堆叠。
      const prev = get().current
      if (prev) prev.resolve(false)
      set({ current: { ...opts, resolve } })
    }),
  resolve: (ok) => {
    const cur = get().current
    if (!cur) return
    cur.resolve(ok)
    set({ current: null })
  },
}))

/** 命令式确认:返回 Promise<boolean>(true=确认,false=取消)。 */
export function confirmDialog(opts: ConfirmOptions): Promise<boolean> {
  return useConfirmStore.getState().open(opts)
}
