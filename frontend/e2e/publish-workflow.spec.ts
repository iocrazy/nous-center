import { test, expect } from 'playwright/test'

// PR-3 把发布弹窗从「节点级 3 步」改成「逐 widget 暴露(WorkflowAppEditor)→ 命名」2 步,
// 旧 spec 断言的「发布为服务 · 步骤 1/3」已不存在。这里对齐新流程。
test('publish a workflow into a service via the granular dialog', async ({ page }) => {
  await page.goto('/workflows')
  await expect(page.getByText(/工作流/i)).toBeVisible({ timeout: 10_000 })

  // Topbar「发布服务」→ 新版 PublishDialog。
  await page.getByRole('button', { name: '发布服务' }).click()
  await expect(page.getByRole('heading', { name: '发布为服务' })).toBeVisible()

  // 第 1 步 = 暴露画布(WorkflowAppEditor)。默认会自动勾选 output 类节点为出参,
  // 所以「下一步」一般已可点;若仍禁用(无 output 节点),DOM 兜底勾一个 widget 行
  // (React Flow 画布里的按钮会被 pane 拦截 Playwright 的可信点击,故用 force/DOM)。
  const next = page.getByRole('button', { name: /下一步/ })
  if (await next.isDisabled()) {
    await page.evaluate(() => {
      const btn = [...document.querySelectorAll('.react-flow__node button')][0] as HTMLButtonElement | undefined
      btn?.click()
    })
  }
  await expect(next).toBeEnabled()
  await next.click()

  // 第 2 步 = 命名 + 发布。
  const ts = Date.now().toString().slice(-8)
  await page.getByPlaceholder(/ltx-drama/).fill(`e2e-pub-${ts}`)
  await page.getByRole('button', { name: '发布服务' }).last().click()

  await page.waitForLoadState('networkidle')
  await page.goto('/services')
  await expect(page.getByText(`e2e-pub-${ts}`)).toBeVisible()
})
