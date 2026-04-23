import { test, expect } from 'playwright/test'

test('publish a workflow into a service via the 3-step dialog', async ({ page }) => {
  // Open a workflow tab. The seed in dev DB ships at least one workflow;
  // if not, this spec needs a fresh wf set up in setup. We rely on the
  // workflow editor's "open" path here.
  await page.goto('/workflows')
  await expect(page.getByText(/工作流/i)).toBeVisible({ timeout: 10_000 })

  // Topbar "发布服务" — the v3 PublishDialog.
  await page.getByRole('button', { name: '发布服务' }).click()
  await expect(page.getByText('发布为服务 · 步骤 1 / 3')).toBeVisible()

  // Step 1 → 2 → 3 (defaults pre-checked).
  await page.getByRole('button', { name: /下一步/ }).click()
  await expect(page.getByText('发布为服务 · 步骤 2 / 3')).toBeVisible()
  await page.getByRole('button', { name: /下一步/ }).click()
  await expect(page.getByText('发布为服务 · 步骤 3 / 3')).toBeVisible()

  const ts = Date.now().toString().slice(-8)
  await page.getByPlaceholder('例如：ltx-drama').fill(`e2e-pub-${ts}`)
  await page.getByRole('button', { name: '发布服务' }).last().click()

  // Either: detail page opens, or the dialog closes without error and the
  // services list shows the new row.
  await page.waitForLoadState('networkidle')
  await page.goto('/services')
  await expect(page.getByText(`e2e-pub-${ts}`)).toBeVisible()
})
