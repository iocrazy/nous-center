// 出图拖到输入(迭代范式,对齐 Infinite-Canvas):生成图回流当下一轮输入。
// 这里是「画廊缩略图(ImageOutputNode)」与「图像输入上传框(ImageUploadWidget)」之间的
// 拖放契约单一真源 —— 纯函数,无 React/DOM 依赖外的副作用,可单测。
//
// 后端 image_input(PR-1 #503)已接受本站签名 URL(/files/images/...),所以前端只需把出图
// URL 原样塞进 node.data.image,无需重新下载/转 base64。

/** 拖放/转换出图 URL 时统一写的 MIME。ImageUploadWidget.onDrop 优先读它。 */
export const NOUS_IMAGE_URL_MIME = 'application/x-nous-image-url'

/** dragstart:把图片 URL 塞进 dataTransfer(自定义 MIME + text/uri-list 双写,
 *  后者让拖到浏览器外/地址栏也带 URL)。 */
export function setImageDrag(e: React.DragEvent, url: string): void {
  e.stopPropagation()
  e.dataTransfer.setData(NOUS_IMAGE_URL_MIME, url)
  e.dataTransfer.setData('text/uri-list', url)
  e.dataTransfer.effectAllowed = 'copy'
}

/** onDrop:从 dataTransfer 取拖进来的图片 URL(优先自定义 MIME,回退 text/uri-list)。
 *  无 URL(比如拖的是本地文件)→ null,调用方回退走 File 读取。 */
export function readImageDropUrl(dt: DataTransfer): string | null {
  const raw = dt.getData(NOUS_IMAGE_URL_MIME) || dt.getData('text/uri-list')
  const url = raw.trim()
  // text/uri-list 规范允许多行 + # 注释行;取第一条非注释 URL。
  if (!url) return null
  const first = url.split(/\r?\n/).find((l) => l && !l.startsWith('#'))
  return first ? first.trim() : null
}

/** image_input 上传框是否「有图可显示」:base64 data URI、本站绝对路径(/files/...)
 *  或 http(s) URL 都算。空串/undefined → false(显示上传提示)。 */
export function isDisplayableImageValue(value: string | undefined | null): boolean {
  return (
    !!value &&
    (value.startsWith('data:') || value.startsWith('/') || value.startsWith('http'))
  )
}
