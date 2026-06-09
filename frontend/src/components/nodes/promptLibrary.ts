// 提示库(对齐 Infinite-Canvas PROMPT 卡的「提示库」)。
// 模板内容参考 Infinite-Canvas/static/system-prompts/infinite-canvas-prompt-templates.md,
// 精简为通用可直接套用的本地常量。纯数据 + 纯函数,便于单测。

export interface PromptTemplate {
  id: string
  name: string
  /** 适用场景一句话,下拉项副标题。 */
  hint: string
  /** 正向提示词(点选后填入节点)。 */
  prompt: string
}

export const PROMPT_DEFAULT_MAX = 20000

export const PROMPT_TEMPLATES: PromptTemplate[] = [
  {
    id: 'multi-angle-grid',
    name: '多机位九宫格',
    hint: '同一主体 9 个机位,角色多角度参考',
    prompt:
      'A multi-camera angle reference sheet in 3x3 grid layout, showing [主体] from 9 different perspectives (front, 3/4, side, low/high angle, back, top-down). Consistent lighting, uniform warm gray background, soft natural edge transition, professional studio photography, character consistency across all angles, absolutely no text or numbers anywhere.',
  },
  {
    id: 'cinematic-portrait',
    name: '电影感人像',
    hint: '胶片质感、柔光、浅景深人像',
    prompt:
      'Cinematic portrait of [主体], 85mm lens, shallow depth of field, soft window light, medium format film aesthetic, fine organic film grain, muted natural color grade, gentle shadows, no oversharpening, photorealistic skin texture.',
  },
  {
    id: 'product-clean',
    name: '产品净底图',
    hint: '电商主图,干净背景 + 柔影',
    prompt:
      'Studio product shot of [产品], centered, seamless light gray background, soft diffused lighting, subtle contact shadow, crisp focus, high detail materials, commercial e-commerce hero image, no props no text.',
  },
  {
    id: 'storyboard-4',
    name: '剧情四宫格',
    hint: '同一事件 4 个连续阶段/情绪递进',
    prompt:
      'A 2x2 storyboard of [事件], four consecutive moments showing emotional progression, consistent character and environment across frames, cinematic framing, thin clean dividers, no captions no numbers.',
  },
  {
    id: '360-pano',
    name: '360 全景',
    hint: '等距圆柱投影全景场景',
    prompt:
      'Equirectangular 360 panorama of [场景], seamless wraparound, even exposure across the full frame, natural perspective, high dynamic range, photorealistic, no stitching seams no text.',
  },
  {
    id: 'flat-illustration',
    name: '扁平插画',
    hint: '矢量风扁平插画,品牌配图',
    prompt:
      'Flat vector illustration of [主题], bold simple shapes, limited harmonious color palette, clean negative space, subtle grain texture, modern editorial style, no gradients heavy, no text.',
  },
]

/** 计数文案:「92 / 20,000」(千分位)。纯函数,供节点角标 + 单测复用。 */
export function formatCharCount(len: number, max: number = PROMPT_DEFAULT_MAX): string {
  return `${len.toLocaleString('en-US')} / ${max.toLocaleString('en-US')}`
}

/** 是否超出上限(角标变红 + 阻断提交判定用)。 */
export function isOverLimit(len: number, max: number = PROMPT_DEFAULT_MAX): boolean {
  return len > max
}

/** 提示库按关键字筛选(匹配名称 / 场景),空查询返回全部。 */
export function filterTemplates(query: string, templates: PromptTemplate[] = PROMPT_TEMPLATES): PromptTemplate[] {
  const q = query.trim().toLowerCase()
  if (!q) return templates
  return templates.filter(
    (t) => t.name.toLowerCase().includes(q) || t.hint.toLowerCase().includes(q),
  )
}
