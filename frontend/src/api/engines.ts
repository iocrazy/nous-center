import { useQuery, useMutation, useQueryClient, keepPreviousData } from '@tanstack/react-query'
import { apiFetch } from './client'
import { useToastStore } from '../stores/toast'
import { useLiveChannel } from './useLiveChannel'

// round3 #5:useEngines 在 Dashboard / Models / CreateServiceDialog 等多处同时挂载,
// 每个 consumer 的 onMessage 都会跑 → 同一条 model_status 事件弹多条相同 toast。
// 用模块级签名去重:相同 (model:status:detail) 在 1.5s 内只弹一次(invalidate 仍每个跑,
// 那是幂等的)。
let _lastEngineToast = { sig: '', at: 0 }
function _engineToastOnce(sig: string, msg: string, kind: 'success' | 'error') {
  const now = Date.now()
  if (sig === _lastEngineToast.sig && now - _lastEngineToast.at < 1500) return
  _lastEngineToast = { sig, at: now }
  useToastStore.getState().add(msg, kind)
}

export interface EngineInfo {
  name: string
  display_name: string
  type: string
  status: 'loaded' | 'unloaded' | 'loading' | 'failed'
  gpu: number | number[]
  vram_gb: number
  resident: boolean
  /** 统一引擎库:目录条目种类。model=整模型/引擎(可独立加载) upscale=SeedVR2 等 by-key
   *  超分(可独立加载,load 接入在 PR-3) component=单文件组件(随 pipeline 加载,不独立可加载)
   *  lora=LoRA(随模型加载)。缺省 model(向后兼容)。 */
  kind?: 'model' | 'upscale' | 'component' | 'lora'
  /** 已加载单文件组件的 L1 身份串(file|device|dtype|loras,含真实 device)。常驻 toggle 按它
   *  精确匹配,避 device='auto' 错配。未加载 / 非组件 → null。组件 L1 PR-3a。 */
  state_key?: string | null
  local_path: string | null
  local_exists: boolean
  // Remote metadata
  organization: string | null
  model_size: string | null
  frameworks: string[] | null
  libraries: string[] | null
  license: string | null
  languages: string[] | null
  tags: string[] | null
  tensor_types: string[] | null
  description: string | null
  has_metadata: boolean
  auto_detected: boolean
  /**
   * False = the model was discovered on disk but no adapter is wired up
   * (image / video diffusers right now). UI must disable the load
   * button — the backend will 422 with a config hint anyway, but it's
   * cleaner to gate the button than to let users click a doomed action.
   */
  has_adapter: boolean
  loaded_gpu: number | null
  loaded_gpus: number[] | null
  status_detail: string | null
  /** image engines only: how many LoRAs the adapter knows about (loaded
   * value when the model is loaded, scanner total when unloaded). null
   * for non-image engines. */
  lora_count: number | null
}

/**
 * Subscribe to /ws/models and invalidate the ['engines'] query family.
 * Pure sync — no toasts. Use from canvas dropdowns that render inside
 * pages where useEngines() isn't mounted but still need live updates.
 *
 * The shared channel is URL-deduped so calling this from many sites
 * doesn't multiply socket connections.
 */
export function useEnginesLiveSync(): void {
  const qc = useQueryClient()
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const url = `${proto}//${window.location.host}/ws/models`
  useLiveChannel(url, {
    onMessage: (data) => {
      if (data.event === 'model_status') {
        qc.invalidateQueries({ queryKey: ['engines'] })
      }
    },
    onReconnect: () => qc.invalidateQueries({ queryKey: ['engines'] }),
  })
}

export function useEngines() {
  const qc = useQueryClient()
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const url = `${proto}//${window.location.host}/ws/models`

  useLiveChannel(url, {
    onMessage: (data) => {
      if (data.event !== 'model_status') return
      qc.invalidateQueries({ queryKey: ['engines'] })
      const sig = `${data.model}:${data.status}:${data.detail ?? ''}`
      if (data.status === 'loaded') {
        _engineToastOnce(sig, `${data.model} ${data.detail || '加载完成'}`, 'success')
      } else if (data.status === 'failed') {
        _engineToastOnce(sig, `${data.model} 加载失败: ${data.detail}`, 'error')
      } else if (data.status === 'installed') {
        _engineToastOnce(sig, `${data.model} 依赖安装完成`, 'success')
      } else if (data.status === 'install_failed') {
        _engineToastOnce(sig, `${data.model} 依赖安装失败: ${data.detail}`, 'error')
      }
    },
    onReconnect: () => qc.invalidateQueries({ queryKey: ['engines'] }),
  })

  return useQuery({
    queryKey: ['engines'],
    queryFn: () => apiFetch<EngineInfo[]>('/api/v1/engines'),
    // Every status transition arrives via /ws/models; the periodic
    // refetch is a safety net (60s) for the rare case the socket is
    // wedged in some half-open state the browser hides from us.
    refetchInterval: (query) => query.state.status === 'error' ? 10_000 : 60_000,
    refetchOnWindowFocus: false,
    retry: false,
    // Show last-known engines instantly when navigating back to the page;
    // background refetch keeps them fresh. Backend serves cached body in <50ms
    // when warm, so the visible flicker window collapses.
    placeholderData: keepPreviousData,
    staleTime: 30_000,
  })
}

export function useLoadEngine() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) =>
      apiFetch(`/api/v1/engines/${name}/load`, { method: 'POST' }),
    onSuccess: (_, name) => {
      // Immediately invalidate to show "loading" status; terminal toast comes from WebSocket
      qc.invalidateQueries({ queryKey: ['engines'] })
      useToastStore.getState().add(`${name} 开始加载...`, 'info')
    },
    onError: (error: Error) => {
      useToastStore.getState().add(`加载失败: ${error.message}`, 'error')
    },
  })
}

/** 统一引擎库 PR-3:从引擎库预热 SeedVR2(by-key,默认配置)。name='seedvr2:<filename>'。
 *  loaded 状态经 runner Pong 反映(几秒后 engines 刷新出 loaded)。 */
export function usePreloadSeedvr2() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) =>
      apiFetch('/api/v1/engines/seedvr2/preload', {
        method: 'POST', body: JSON.stringify({ name }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['engines'] })
      useToastStore.getState().add('SeedVR2 开始加载...（几秒后引擎库刷新显示常驻）', 'info')
    },
    onError: (error: Error) => {
      useToastStore.getState().add(`SeedVR2 预热失败: ${error.message}`, 'error')
    },
  })
}

export function useUnloadSeedvr2() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) =>
      apiFetch('/api/v1/engines/seedvr2/unload', {
        method: 'POST', body: JSON.stringify({ name }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['engines'] })
      useToastStore.getState().add('SeedVR2 已卸载', 'success')
    },
    onError: (error: Error) => {
      useToastStore.getState().add(`卸载失败: ${error.message}`, 'error')
    },
  })
}

/** SeedVR2 常驻 toggle(组件 L1 PR-2c:by-key 模型常驻 pin)。 */
export function useSetSeedvr2Resident() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ name, resident }: { name: string; resident: boolean }) =>
      apiFetch('/api/v1/engines/seedvr2/resident', {
        method: 'POST', body: JSON.stringify({ name, resident }),
      }),
    onSuccess: (_d, v) => {
      qc.invalidateQueries({ queryKey: ['engines'] })
      useToastStore.getState().add(v.resident ? 'SeedVR2 已设为常驻' : 'SeedVR2 已取消常驻', 'success')
    },
    onError: (error: Error) => {
      useToastStore.getState().add(`常驻切换失败: ${error.message}`, 'error')
    },
  })
}

/** 单组件预加载到显存 + 可选常驻(组件 L1 PR-2a)。name='component:<kind>:<path>';dtype 选精度。 */
export function usePreloadComponent() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ name, dtype, device, resident }:
      { name: string; dtype?: string; device?: string; resident?: boolean }) =>
      apiFetch('/api/v1/engines/component/preload', {
        method: 'POST',
        body: JSON.stringify({
          name, dtype: dtype ?? 'bfloat16',
          ...(device ? { device } : {}),  // 省略 → 后端 auto 自动选卡
          resident: resident ?? false,
        }),
      }),
    onSuccess: (_d, v) => {
      qc.invalidateQueries({ queryKey: ['engines'] })
      useToastStore.getState().add(
        `组件开始预加载${v.resident ? ' + 常驻' : ''}...（几秒后引擎库刷新显示已加载）`, 'info')
    },
    onError: (error: Error) => {
      useToastStore.getState().add(`组件预加载失败: ${error.message}`, 'error')
    },
  })
}

/** 卸载已预加载组件(出 L1 + 释放显存,统一模型管理收尾 PR-1)。优先 state_key,否则 name+device/dtype。 */
export function useUnloadComponent() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ state_key, name, device, dtype }:
      { state_key?: string | null; name?: string; device?: string; dtype?: string }) =>
      apiFetch('/api/v1/engines/component/unload', {
        method: 'POST',
        body: JSON.stringify(
          state_key ? { state_key } : { name, device, dtype: dtype ?? 'bfloat16' }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['engines'] })
      useToastStore.getState().add('组件开始卸载...（几秒后引擎库刷新）', 'info')
    },
    onError: (error: Error) => {
      useToastStore.getState().add(`组件卸载失败: ${error.message}`, 'error')
    },
  })
}

/** 已加载组件常驻 toggle(组件 L1 PR-2b)。优先用 state_key 精确匹配,否则 name+device/dtype。 */
export function useSetComponentResident() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ state_key, name, device, dtype, resident }:
      { state_key?: string | null; name?: string; device?: string; dtype?: string; resident: boolean }) =>
      apiFetch('/api/v1/engines/component/resident', {
        method: 'POST',
        body: JSON.stringify(
          state_key ? { state_key, resident } : { name, device, dtype: dtype ?? 'bfloat16', resident }),
      }),
    onSuccess: (_d, v) => {
      qc.invalidateQueries({ queryKey: ['engines'] })
      useToastStore.getState().add(v.resident ? '组件已设为常驻' : '组件已取消常驻', 'success')
    },
    onError: (error: Error) => {
      useToastStore.getState().add(`常驻切换失败: ${error.message}`, 'error')
    },
  })
}

export function useUnloadEngine() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) =>
      apiFetch(`/api/v1/engines/${name}/unload?force=true`, { method: 'POST' }),
    onSuccess: (_, name) => {
      qc.invalidateQueries({ queryKey: ['engines'] })
      useToastStore.getState().add(`${name} 已卸载`, 'success')
    },
    onError: (error: Error) => {
      useToastStore.getState().add(`卸载失败: ${error.message}`, 'error')
    },
  })
}

export function useSyncMetadata() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () =>
      apiFetch('/api/v1/engines/sync-metadata', { method: 'POST' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['engines'] }),
  })
}

export function useSetResident() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ name, resident }: { name: string; resident: boolean }) =>
      apiFetch(`/api/v1/engines/${name}/resident?resident=${resident}`, { method: 'PATCH' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['engines'] }),
    onError: (error: Error) => {
      useToastStore.getState().add(`设置失败: ${error.message}`, 'error')
    },
  })
}

export function useScanModels() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () =>
      apiFetch<{
        count: number
        local_available?: number
        not_local?: number
        models: string[]
      }>('/api/v1/engines/scan', { method: 'POST' }),
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ['engines'] })
      // PR-11:旧文案「扫描完成,共 N 个模型」实际是 yaml 配置识别数,会比
      // 引擎库可见(`/api/v1/engines` 过滤 local_path 后)多出未下载的份额 —
      // 用户曾报告「扫到 25 实际只能用 16」之误导。后端 PR-11 同时返回
      // local_available/not_local,有差异时显式拆分提示。
      // 兼容旧后端 payload(无 local_available 字段时降级为原文案)。
      const total = data.count
      const local = data.local_available
      const missing = data.not_local
      const msg =
        local != null && missing != null && missing > 0
          ? `识别 ${total} 个模型 · 本地可用 ${local} · ${missing} 个未下载`
          : `扫描完成,共 ${total} 个模型`
      useToastStore.getState().add(msg, 'success')
    },
    onError: (error: Error) => {
      useToastStore.getState().add(`扫描失败: ${error.message}`, 'error')
    },
  })
}

/** PR-D4:手动卸载所有 image adapter(走 `_models[derived_id]` 统一字典 + 释放显存)。
 * 调用后 dashboard / 引擎库自动看到 image adapter 消失。 */
export function useUnloadImageAdapters() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () =>
      apiFetch<{ unloaded: string[]; count: number }>(
        '/api/v1/engines/unload-image-adapters', { method: 'POST' },
      ),
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ['engines'] })
      qc.invalidateQueries({ queryKey: ['monitor-stats'] })
      const msg = data.count > 0
        ? `已卸载 ${data.count} 个 image adapter`
        : '当前无 image adapter,无需卸载'
      useToastStore.getState().add(msg, 'success')
    },
    onError: (error: Error) => {
      useToastStore.getState().add(`卸载失败: ${error.message}`, 'error')
    },
  })
}

export function useRefreshMetadata() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) =>
      apiFetch(`/api/v1/engines/${name}/refresh-metadata`, { method: 'POST' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['engines'] }),
  })
}

/** Bug 3 PR-2c:runner 子进程里加载的 combo adapter 实体(image/tts)。它们是工作流
 * 动态组装的单文件 combo,不对应注册卡片,所以独立于 EngineInfo,在引擎库「已加载」
 * tab 单独渲染。数据来自 /api/v1/engines/loaded-adapters(聚合各 runner 的 Pong 快照)。 */
export interface LoadedAdapter {
  model_id: string
  model_type: string
  group_id: string
  gpu_index: number | null
  vram_mb: number | null
  pipeline_class: string | null
  source_files: string[]
  display_name: string
  last_used_ago_sec: number | null
}

export function useLoadedAdapters() {
  return useQuery({
    queryKey: ['loaded-adapters'],
    queryFn: () =>
      apiFetch<{ count: number; entries: LoadedAdapter[] }>(
        '/api/v1/engines/loaded-adapters',
      ),
    // 后端快照由 runner 节点完成时即时 reconcile(PR-2b);这里 8s 兜底轮询。
    refetchInterval: 8000,
    refetchOnWindowFocus: false,
    retry: false,
    staleTime: 4000,
  })
}

export interface GpuDevice {
  index: number
  name: string
  vram_gb: number
}

export function useGpus() {
  return useQuery({
    queryKey: ['gpus'],
    queryFn: () =>
      apiFetch<{ count: number; devices: GpuDevice[] }>('/api/v1/engines/gpus'),
  })
}

export function useSetGpu() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ name, gpu }: { name: string; gpu: number }) =>
      apiFetch(`/api/v1/engines/${name}/gpu?gpu=${gpu}`, { method: 'PATCH' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['engines'] }),
    onError: (error: Error) => {
      useToastStore.getState().add(`GPU 分配失败: ${error.message}`, 'error')
    },
  })
}
