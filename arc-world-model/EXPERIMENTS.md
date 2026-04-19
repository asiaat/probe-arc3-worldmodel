# arcwm — Experiments log

## Week 1 — Data collection

**Dates:** 2026-04-19

### Setup
- Repo created, 19 tests pass
- SDK verified: obs.frame is FrameDataRaw with frame attribute
- Human baselines: REAL from SDK via env.info.baseline_actions
- Replay buffer: 18,000 transitions on games [ls20, ft09]

### Data rebalancing
- Initial split ls20=8624 / ft09=1376 was skewed (86/14)
- Topped up ft09 to 9376 transitions
- Final buffer: 18,000 transitions total (ls20=8624, ft09=9376)
- Cause: ft09 episodes are much shorter (~60-100 steps) vs ls20 (~400-500 steps)

### Real human baselines (from SDK)
- ft09: baseline_actions = [43, 12, 23, 28, 65, 37] (6 levels)
- ls20: baseline_actions = [22, 123, 73, 84, 96, 192, 186] (7 levels)

### Data observations (from notebook 01)
**Visual content:**
- ls20: geometric puzzles with colored shapes, grid-based transformations
- ft09: different game mechanic, more sparse frames

**Action effects:**
- Action distribution: roughly uniform (~1200 per action for 15 actions)
- State-change rate: 18.8%
- 6-7 actions seem most common (could indicate preferred directions)

**Terminal states:**
- 129 terminals in buffer
- Frames right after terminal action show GAME_OVER state
- No wins in 18K transitions (random policy cannot win)

**Design implications for Week 2:**
- Encoder needs to preserve object shapes and colors
- Most actions do nothing - model must handle sparse rewards
- VAE reconstruction likely works, but may need segmentation-aware losses
- Game-specific dynamics (ls20 vs ft09) suggest multi-game training

### Data quality notes
- Both games well represented in buffer
- Reward values: min=-1.0, max=0.05, mean=-0.0008, nonzero_frac=13.2%
- Terminal states present (129 episodes ended in GAME_OVER)
- Action coverage balanced

### Questions for Week 2
- Encoder should capture object shapes/colors - frames have visible game elements
- Transition model needs to handle variable episode lengths
- ls20 vs ft09 have very different dynamics - consider game-specific modeling?
- What does a "state change detector" need to preserve?