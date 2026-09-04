import type { MenuItem } from '../ui/ContextMenu'
import type { GpuDevice, GpuGroup } from '../../api/engines'

/**
 * 「GPU 分配」子菜单：单卡项 + 分隔线 + 组合项（张量并行）。
 *
 * 组合项来自后端 `GET /api/v1/gpu/groups` —— 它只列**同型号**卡的组合（异构组队做
 * 张量并行会按最小卡算或直接 OOM），NVLink 组排在前面。选中组合发 `{gpus}`，选中
 * 单卡发 `{gpu}`（后端会顺手清掉已有的组 = 退出组）。
 *
 * 抽成纯函数是为了能直接测：整个 ModelsOverlay 渲染起来要 mock 十几个 hook。
 */
export function buildGpuAssignSubmenu(opts: {
  devices: GpuDevice[]
  groups: GpuGroup[]
  /** 当前配置的 GPU（单卡 number，或旧数据里的 number[]） */
  currentGpu: number | number[] | undefined
  /** 当前配置的 GPU 组 */
  currentGroup: number[] | null | undefined
  onPickGpu: (gpu: number) => void
  onPickGroup: (gpus: number[]) => void
}): MenuItem[] {
  const { devices, groups, currentGpu, currentGroup, onPickGpu, onPickGroup } = opts
  const inGroup = !!currentGroup && currentGroup.length > 1

  const singles: MenuItem[] = devices.map((g) => ({
    label: `GPU ${g.index}: ${g.name}`,
    onClick: () => onPickGpu(g.index),
    // 配了组时单卡项一律可点（= 退出组回到单卡）；没配组才按当前卡置灰。
    disabled: inGroup
      ? false
      : Array.isArray(currentGpu)
        ? currentGpu.includes(g.index)
        : currentGpu === g.index,
  }))

  if (groups.length === 0) return singles

  const combos: MenuItem[] = groups.map((grp) => {
    const cur = currentGroup ?? []
    const isCurrent =
      cur.length === grp.gpus.length && cur.every((v, i) => v === grp.gpus[i])
    return {
      label:
        `组合 GPU ${grp.gpus.join('+')}` +
        `（${grp.nvlink ? 'NVLink' : 'PCIe'} · ${grp.total_gb}GB）`,
      onClick: () => onPickGroup(grp.gpus),
      disabled: isCurrent,
    }
  })

  return [...singles, { label: '', divider: true }, ...combos]
}
