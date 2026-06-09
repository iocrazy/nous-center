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
