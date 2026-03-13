// src/utils/uid.ts
export function uid(): string {
  return crypto.randomUUID().slice(0, 8)
}
