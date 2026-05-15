import { describe, it, expect, beforeEach } from 'vitest'
import { useExecutionStore } from './execution'

describe('execution store — task icon badge', () => {
  beforeEach(() => {
    useExecutionStore.setState({ taskIconBadge: 0, taskPanelOpen: false })
  })

  it('bumpTaskBadge increments the badge count', () => {
    useExecutionStore.getState().bumpTaskBadge()
    useExecutionStore.getState().bumpTaskBadge()
    expect(useExecutionStore.getState().taskIconBadge).toBe(2)
  })

  it('clearTaskBadge resets to zero', () => {
    useExecutionStore.setState({ taskIconBadge: 5 })
    useExecutionStore.getState().clearTaskBadge()
    expect(useExecutionStore.getState().taskIconBadge).toBe(0)
  })
})
