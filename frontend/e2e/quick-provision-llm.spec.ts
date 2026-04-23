import { test, expect } from 'playwright/test'

test('quick-provision an LLM service and land on its detail page', async ({ page }) => {
  await page.goto('/services')
  await expect(page.getByRole('heading', { name: '服务' })).toBeVisible()

  await page.getByRole('button', { name: /快速开通/ }).click()
  await expect(page.getByRole('heading', { name: '快速开通服务' })).toBeVisible()

  // Pick LLM (default), fill name, leave engine default → should be the first
  // option in the engines select (the test backend must have at least one).
  const engineSelect = page.getByRole('combobox').first()
  await engineSelect.selectOption({ index: 1 })

  const ts = Date.now().toString().slice(-8)
  await page.getByPlaceholder('例如：qwen-chat').fill(`e2e-llm-${ts}`)

  await page.getByRole('button', { name: '开通' }).click()

  // Land on detail page; tabs visible.
  await expect(page.getByRole('heading', { level: 1 })).toContainText(`e2e-llm-${ts}`)
  await expect(page.getByText('Playground', { exact: false })).toBeVisible()
})
