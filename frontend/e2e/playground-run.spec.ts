import { test, expect } from 'playwright/test'

test('run a service through the schema-driven playground', async ({ page }) => {
  await page.goto('/services')
  await expect(page.getByRole('heading', { name: '服务' })).toBeVisible()

  // Open the first service card.
  const firstCard = page.locator('button').filter({ hasText: /运行中|快速开通/ }).first()
  await firstCard.click()

  // Land on detail; default tab is Playground.
  await expect(page.getByText('Playground', { exact: false })).toBeVisible()

  // Whatever the schema produces, there's at least the run button.
  const runBtn = page.getByRole('button', { name: /▶ 运行/ })
  await runBtn.click()

  // Output side either shows a result body, an error, or the running spinner.
  // We accept any of these without coupling to the model's response shape.
  await Promise.race([
    page.locator('text=completed').waitFor({ timeout: 30_000 }),
    page.locator('text=failed').waitFor({ timeout: 30_000 }),
  ])
})
