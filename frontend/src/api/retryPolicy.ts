import { NousApiError } from './errors'

/** 默认重试次数(与 React Query 自身的默认一致)。 */
export const MAX_RETRIES = 3

/**
 * React Query 的全局重试判定。
 *
 * React Query 默认对**任何**失败重试 3 次(共 4 次请求)。这对临时性故障
 * (5xx / 断网 / 超时)是对的,但对 4xx **永远没有意义** —— 4xx 的语义就是
 * 「请求本身不对」(资源不存在、参数非法、无权限),再发一百遍还是同样的错。
 *
 * 后果不是多几个请求那么轻:失败要等满 4 轮才显形,这期间界面一直显示
 * 「加载中…」,用户以为在正常加载。实测用一个不存在的 id 打服务详情页,
 * 要 **约 10 秒** 才看到错误(2026-08-31)。
 *
 * 这也是为什么仓库里有十来处 `retry: false` 的手写补丁 —— 之前是一处处
 * 绕过。改默认之后那些补丁仍然生效(每查询设置优先),但不再是必需品。
 */
export function shouldRetry(failureCount: number, error: unknown): boolean {
  const status = httpStatusOf(error)
  // 4xx = 客户端错误,重试改变不了结果。5xx / 网络错误 / 未知错误照旧重试。
  if (status !== undefined && status >= 400 && status < 500) return false
  return failureCount < MAX_RETRIES
}

/** 从错误里取 HTTP 状态码;不是 HTTP 错误(如断网 TypeError)时返回 undefined。 */
function httpStatusOf(error: unknown): number | undefined {
  if (error instanceof NousApiError) return error.httpStatus
  // 防御:跨 bundle / 测试替身可能不是同一个类实例,退回鸭子类型。
  const s = (error as { httpStatus?: unknown } | null)?.httpStatus
  return typeof s === 'number' ? s : undefined
}
