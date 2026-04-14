export class NousApiError extends Error {
  type: string
  code?: string
  param?: string
  requestId?: string
  httpStatus: number

  constructor(payload: any, httpStatus: number, fallbackRequestId?: string) {
    const err = payload?.error ?? {}
    super(err.message ?? `HTTP ${httpStatus}`)
    this.name = 'NousApiError'
    this.type = err.type ?? 'api_error'
    this.code = err.code
    this.param = err.param
    this.requestId = err.request_id ?? fallbackRequestId
    this.httpStatus = httpStatus
  }
}
