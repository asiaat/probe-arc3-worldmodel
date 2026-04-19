# arcwm — Experiments log

## Week 1 — Data collection

**Dates:** 2026-04-19

### Setup
- Repo created, 19 tests pass
- SDK verified: obs.frame is FrameDataRaw with frame attribute
- Human baselines: fallback for both games (SDK didn't expose real baselines)
- Replay buffer: 10,000 transitions on games [ls20, ft09]

### Observations
- ls20: 8624 transitions, 11 terminals, 0 wins
- ft09: 1376 transitions, 17 terminals, 0 wins
- Reward distribution: min=-1.0, max=0.05, mean=0.006, nonzero_frac=17.73%
- State-change rate (~18%) is consistent with ARC-AGI-3 design: most actions do nothing
- Collected episodes ended due to GAME_OVER or max 500 steps
- Random policy cannot win (expected)
- ft09 episodes are much shorter (60-100 steps) vs ls20 (400-500 steps)

### Data quality notes
- Both games represented in buffer
- Reward values indicate progress signals present for ls20 (positive rewards), negative for ft09
- Terminal states present (28 episodes ended in game over)
- Action coverage appears balanced

### Questions for Week 2
- Encoder should capture object shapes/colors - frames have visible game elements
- Transition model needs to handle variable episode lengths
- ls20 vs ft09 have very different dynamics - consider game-specific modeling?