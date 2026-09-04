import type { MenuItem } from '../ui/ContextMenu'
import type { GpuDevice, GpuGroup } from '../../api/engines'

/**
 * 「GPU 分配」子菜单：单卡项 + 分隔线 + 组合项（张量并行）。
 *
 * 组合项来自后端 `GET /api/v1/gpu/groups` —— 那是 `configs/hardware.yaml` 里**声明过**
 * 的多卡 group，不是前端枚举出来的组合（yaml 记着"哪张卡在驱动显示器"这类运维约束）。
 * yaml 没声明多卡组 → `groups` 为空 → 只剩单卡项，与改动前一样。
 *
 * 组合项只对 `supports_gpu_group` 的引擎显示：只有 vLLM / SGLang 这类子进程型 LLM
 * 适配器能跨卡；给别的引擎配组，后端会按两张卡各预留一半而适配器实际单卡跑 ——
 * 幻影占用 + UI 撒谎，所以这里根本不给点。
 *
 * 选中组合发 `{gpus}`，选中单卡发 `{gpu}`（后端会显式清空组 = 退出组）。
 * 判"当前是不是组"**只看 `gpus`**：`gpu` 永远是主卡 int。
 *
 * 抽成纯函数是为了能直接测：整个 ModelsOverlay 渲染起来要 mock 十几个 hook。
 */
export function buildGpuAssignSubmenu(opts: {
  devices: GpuDevice[]
  groups: GpuGroup[]
  /** 当前配置的主卡 */
  currentGpu: number | undefined
  /** 当前配置的 GPU 组（null / undefined = 单卡） */
  currentGroup: number[] | null | undefined
  /** 该引擎的适配器是否接受 GPU 组 */
  supportsGroup?: boolean
  onPickGpu: (gpu: number) => void
  onPickGroup: (gpus: number[]) => void
}): MenuItem[] {
  const {
    devices, groups, currentGpu, currentGroup, supportsGroup = false,
    onPickGpu, onPickGroup,
  } = opts
  const inGroup = !!currentGroup && currentGroup.length > 1

  const singles: MenuItem[] = devices.map((g) => ({
    label: `GPU ${g.index}: ${g.name}`,
    onClick: () => onPickGpu(g.index),
    // 在组里时单卡项一律可点（= 退出组回到单卡）；不在组里才按当前主卡置灰。
    disabled: inGroup ? false : currentGpu === g.index,
  }))

  if (!supportsGroup || groups.length === 0) return singles

  const combos: MenuItem[] = groups.map((grp) => {
    const cur = currentGroup ?? []
    const isCurrent =
      cur.length === grp.gpus.length && cur.every((v, i) => v === grp.gpus[i])
    const warn = grp.display_gpus && grp.display_gpus.length > 0
      ? ` ⚠ GPU ${grp.display_gpus.join('+')} 在驱动显示器`
      : ''
    return {
      label:
        `组合 GPU ${grp.gpus.join('+')}` +
        `（${grp.nvlink ? 'NVLink' : 'PCIe'} · ${grp.total_gb}GB）${warn}`,
      onClick: () => onPickGroup(grp.gpus),
      disabled: isCurrent,
    }
  })

  return [...singles, { label: '', divider: true }, ...combos]
}
