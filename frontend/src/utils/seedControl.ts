/**
 * ComfyUI 风格的 control_after_generate —— 每次 Run 完成后按模式更新 KSampler 的 seed。
 * 纯前端(对齐 ComfyUI src/scripts/valueControl.ts:nextValueForLinkedTarget 的数值分支)。
 *
 * ComfyUI 行为:fixed=不变 / increment=+step / decrement=-step / randomize=随机,
 * seed step=1,范围 [0, 2^64)。JS Number 只能安全表示到 2^53-1,所以 randomize 用
 * 安全范围(seed 多样性足够),increment/decrement 用 BigInt 兜大 seed(用户截图里
 * seed=99611110155462212 > 2^53,直接 +1 会丢精度)。
 */

export type SeedControlMode = 'fixed' | 'increment' | 'decrement' | 'randomize'

const SEED_MIN = 0n
// ComfyUI seed max = 0xffffffffffffffff(2^64-1)。randomize 用 2^53-1 安全范围(seed
// 多样性绰绰有余,且 < Number.MAX_SAFE_INTEGER 不丢精度);increment/decrement 仍可达 2^64。
const SEED_MAX = (1n << 64n) - 1n
const SAFE_RANDOM_MAX = Number.MAX_SAFE_INTEGER // 2^53 - 1

function clamp(v: bigint): bigint {
  if (v < SEED_MIN) return SEED_MIN
  if (v > SEED_MAX) return SEED_MAX
  return v
}

/**
 * 给定当前 seed(字符串,可能是大整数)+ 模式,返回下一个 seed 字符串。
 * 当前 seed 解析失败(空/非数字)时 randomize/increment 从 0 起算。
 */
export function nextSeed(currentSeed: string | number | null | undefined, mode: SeedControlMode): string {
  if (mode === 'fixed') {
    return String(currentSeed ?? '')
  }
  if (mode === 'randomize') {
    // 用 crypto 拿高质量随机(Math.random 也行,这里求稳)。范围 [0, 2^53)。
    const r = Math.floor(Math.random() * SAFE_RANDOM_MAX)
    return String(r)
  }
  // increment / decrement —— BigInt 兜大 seed
  let cur: bigint
  try {
    const s = String(currentSeed ?? '').trim()
    cur = s === '' ? 0n : BigInt(s)
  } catch {
    cur = 0n
  }
  const next = mode === 'increment' ? cur + 1n : cur - 1n
  return String(clamp(next))
}
