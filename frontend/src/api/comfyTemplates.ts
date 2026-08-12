import { apiFetch } from './client'

// ---------- shared types ----------

export interface ComfyExposedParam {
  key: string
  label?: string
  type?: string
  default?: unknown
  min?: number
  max?: number
  step?: number
  options?: unknown[]
  required?: boolean
  random?: boolean
  comfy_node_id: string
  comfy_input: string
}

export interface ComfyTemplateRow {
  id: string
  name: string
  service_name: string | null
  node_count: number
  exposed_count: number
}

export interface ComfyTemplateDetail {
  id: string
  name: string
  service_name: string
  workflow_json: Record<string, unknown>
  exposed_params: ComfyExposedParam[]
}

export interface CreateComfyTemplateResult {
  id: string
  name: string
  service_name: string
  node_count: number
}

export interface ComfyDevice {
  name: string
  type: string
  index: number
  vram_total: number
  vram_free: number
  vram_used: number
  torch_vram_total: number
}

export interface ComfyHealth {
  online: boolean
  queue_depth: number
  version: string
  base_url: string
  timeout_s: number
  devices?: ComfyDevice[]
}

export interface FreeComfyVramResult {
  ok: boolean
  /** 显存是否已真降下来。ComfyUI 的 /free 是异步的(只设 flag,worker 唤醒后才卸载),
   *  后端轮询至多 6s;超时仍未降则 false —— 此时 UI 该说"已触发,可能仍在进行"。 */
  settled: boolean
  /** 本次实际归还的字节数(未落定时为 0)。 */
  freed_bytes: number
  devices: ComfyDevice[]
}

// ---------- calls ----------

export function createComfyTemplate(
  name: string,
  workflow: Record<string, unknown>,
): Promise<CreateComfyTemplateResult> {
  return apiFetch<CreateComfyTemplateResult>('/api/v1/comfy-templates', {
    method: 'POST',
    body: JSON.stringify({ name, workflow }),
  })
}

export function putMapping(
  id: string | number,
  exposedParams: ComfyExposedParam[],
): Promise<{ ok: boolean }> {
  return apiFetch<{ ok: boolean }>(`/api/v1/comfy-templates/${id}/mapping`, {
    method: 'PUT',
    body: JSON.stringify({ exposed_params: exposedParams }),
  })
}

export function getComfyTemplate(id: string | number): Promise<ComfyTemplateDetail> {
  return apiFetch<ComfyTemplateDetail>(`/api/v1/comfy-templates/${id}`)
}

export function listComfyTemplates(): Promise<ComfyTemplateRow[]> {
  return apiFetch<ComfyTemplateRow[]>('/api/v1/comfy-templates')
}

export function reuploadComfyTemplate(
  id: string | number,
  workflow: Record<string, unknown>,
): Promise<{ stale_keys: string[] }> {
  return apiFetch<{ stale_keys: string[] }>(`/api/v1/comfy-templates/${id}`, {
    method: 'PUT',
    body: JSON.stringify({ workflow }),
  })
}

export function getComfyHealth(): Promise<ComfyHealth> {
  return apiFetch<ComfyHealth>('/api/v1/comfy/health')
}

export function getObjectInfo(): Promise<Record<string, unknown>> {
  return apiFetch<Record<string, unknown>>('/api/v1/comfy/object-info')
}

export function freeComfyVram(): Promise<FreeComfyVramResult> {
  return apiFetch<FreeComfyVramResult>('/api/v1/comfy/free', { method: 'POST' })
}
