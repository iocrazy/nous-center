/** 页面 overlay 懒加载 chunk 到达前的占位层。
 *
 *  必须是**不透明满屏**的:overlay 全是 `position:absolute; inset:0; background:var(--bg)`,
 *  盖在工作流画布上面。之前这里的 Suspense fallback 是 `null` —— null 不是「什么都不显示」,
 *  而是把底下的画布整个露出来,于是刷新 /services /settings 会先闪一帧画布。
 *
 *  文字用 CSS delay 淡入(不是 JS 定时器):底色立刻铺满,`加载中…` 只在 chunk
 *  超过 ~180ms 才浮现,秒开的情况看起来就是直接出页面,不会闪一下 loading 框。
 */
export default function OverlayLoading() {
  return (
    <div
      data-testid="overlay-loading"
      className="absolute inset-0 z-[16] flex items-center justify-center"
      style={{ background: 'var(--bg)' }}
      aria-busy="true"
      aria-live="polite"
    >
      <span
        className="text-sm"
        style={{
          color: 'var(--muted)',
          animation: 'nous-delayed-fade-in 160ms ease-out 180ms both',
        }}
      >
        加载中…
      </span>
    </div>
  )
}
