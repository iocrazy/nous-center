import '@testing-library/jest-dom/vitest'

// React Flow(@xyflow/react)在 jsdom 里需要 ResizeObserver + matchMedia;
// jsdom 默认不提供。补 stub,让含画布的组件(WorkflowAppEditor 等)能在单测里挂载。
if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver
}

if (typeof globalThis.matchMedia === 'undefined') {
  globalThis.matchMedia = ((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  })) as unknown as typeof globalThis.matchMedia
}

// zustand 的 persist 中间件在 store 模块**导入期**就抓 `localStorage`
// (workspace.ts:484 `createJSONStorage(() => localStorage)`)。这个 jsdom 环境里
// 它可能是 undefined —— 于是任何 setState 都炸 "Cannot read properties of
// undefined (reading 'setItem')"。
//
// 之前没暴露是因为**顺序依赖**:全量跑时碰巧有别的文件先把它准备好了。文件一多、
// vitest 的 file→worker 分配一变,workspace.test.ts / NodePropertyPanel.test.tsx
// 就整组红(单跑这两个文件在任何分支上都必红)。
// 和上面 ResizeObserver / matchMedia 同样处理:缺就补一个内存实现。
if (typeof globalThis.localStorage === 'undefined') {
  const mem = new Map<string, string>()
  globalThis.localStorage = {
    getItem: (k: string) => (mem.has(k) ? mem.get(k)! : null),
    setItem: (k: string, v: string) => void mem.set(k, String(v)),
    removeItem: (k: string) => void mem.delete(k),
    clear: () => mem.clear(),
    key: (i: number) => Array.from(mem.keys())[i] ?? null,
    get length() {
      return mem.size
    },
  } as unknown as Storage
}
