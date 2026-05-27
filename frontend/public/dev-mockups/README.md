# dev-mockups · 视觉真相参考

本目录的 HTML 是 **dev-only design reference**,作用是给后续 React 实现做「视觉真相锚定」 —
React 组件 port 时,本地起 `npm run dev` 后,在浏览器并排打开:

- tab 1:`http://localhost:9999/dev-mockups/task-panel.html`(本目录的 mockup 真相)
- tab 2:实际 React 应用(`http://localhost:9999/`)

肉眼对比 + `/design-review` skill 自动 audit。SSIM 不到 0.95 不算视觉通过。

## 为什么放在 `public/` 而不是 `src/`

- `public/` 下的文件 vite dev/prod 都直接 serve,URL 路径可预测
- prod build 会包含本 HTML(~40KB),但这是 dev reference 的代价 —— admin 不会访问,bundle 影响可忽略
- 后续如果想 prod 严格排除,可以在 vite.config.ts 加 build filter,目前先 keep simple

## 文件清单

| 文件 | 来源 | 用途 |
|---|---|---|
| `task-panel.html` | `docs/superpowers/specs/assets/2026-05-27-task-panel-reset/variant-final.html` | task panel reset 终极 mockup(三态交互一图全显示) |

## 同步规则

**source of truth = `docs/superpowers/specs/assets/2026-05-27-task-panel-reset/variant-final.html`**。

如果 spec 修订(增加 variant / 修改设计 token / 改交互),按这个顺序:

1. 改 `docs/superpowers/specs/assets/.../variant-final.html`(或新增 `variant-v2.html`)
2. `cp docs/.../variant-final.html frontend/public/dev-mockups/task-panel.html` 同步
3. 同步 `frontend/src/styles/task-panel-tokens.css` 里的 CSS variable(如果 token 变了)
4. 同 PR commit 一起 push,确保 spec / mockup / token 三处永远一致

后续可以加 pre-commit hook 校验 spec assets 跟 public/dev-mockups 文件一致。

## 相关 token

`frontend/src/styles/task-panel-tokens.css` 是从本 HTML 抽出的 CSS variable 集合 ——
**React 组件用 token,不用 HTML 里的 hardcoded color**。HTML 是给人眼看的快照,
token 才是给代码用的真相。
