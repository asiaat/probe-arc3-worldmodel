# WEEK_1_COMPLETION.md — Finish Week 1 Before Moving to Week 2

**Status from last report:** ~85% complete.
- ✅ 19 tests pass
- ✅ 10K transitions collected (but imbalanced: ls20=8624, ft09=1376)
- ✅ EXPERIMENTS.md has Week 1 entry
- ⚠️ Human baselines are fallback placeholders, not real SDK values
- ❌ `notebooks/00_playing_notes.md` missing (human hasn't played yet)
- 🤔 `01_sdk_exploration.ipynb` exists — unclear if frames were actually viewed

**Your task:** Close these four gaps before Week 2. Estimated ~3 hours of
work total. Some can run unattended.

---

## Task priority (do in this order)

### Priority 1 — Data rebalancing (~1 hour, mostly unattended)

**Problem:** 86% of transitions are from `ls20`, only 14% from `ft09`.
This imbalance will make the world model specialise for `ls20` dynamics
and generalise poorly.

**Root cause hypothesis:** `ft09` has much shorter episodes (faster
GAME_OVER) than `ls20`. Alternating episodes produces very unequal
transition counts.

**Fix:** Top up `ft09` until the buffer has at least 8000 `ft09`
transitions. Run:

```bash
# Get current counts before starting
make buffer-stats

# Collect more ft09 data specifically
python scripts/collect_data.py \
    --games ft09 \
    --target 18000 \
    --max-steps 500 \
    --output-dir ./data/replay
```

Note: `--target` is the **total buffer size goal**, not new transitions.
Since buffer currently has ~10K and we want ft09 to grow by ~6600, set
target to ~16600 and run with `--games ft09` only. This will keep adding
ft09 episodes until total reaches 16600.

**Verify after:**
```bash
make buffer-stats
```

Expected per_game: `ls20 ~= 8624`, `ft09 >= 8000`.

If `ft09` episodes are very short (< 30 steps average), this may take a
while — be patient. Ctrl+C safe, resumable.

**Success criterion:** Both games have at least 8000 transitions each.
Document in `EXPERIMENTS.md`:
```markdown
### Data rebalancing
Initial split ls20=8624 / ft09=1376 was skewed (86/14). Topped up
ft09 to XXXX transitions. Final buffer: XXX transitions total.
Cause: ft09 episodes average ~N steps vs ls20 ~M steps.
```

---

### Priority 2 — Play the games (30 minutes, mandatory)

**Go to https://three.arcprize.org and play at least 2-3 games.**
Write notes in `notebooks/00_playing_notes.md`.

This is NOT optional. The best RL practitioners play their own
benchmarks. Without this, you'll make bad design decisions in Week 2
because you won't understand what the agent is trying to do.

Create the file and answer these questions concretely:

```markdown
# Playing notes — ARC-AGI-3 human experience

## Games played
- Game 1: [name, date/time, did you win?]
- Game 2: ...
- Game 3: ...

## Figuring out actions
How did you learn what the 6 actions do? What did you try first?
Did some actions seem to do nothing? Did clicking matter?

## Progress signals
What visual cues told you "that action did something useful"?
When you made a mistake, how did you know?

## Recognising wins
What did it feel like to win a level? Was the transition obvious
or subtle? Did the game tell you explicitly?

## Estimated action efficiency
Roughly how many actions did each level take you?
If you played the same level twice, did you get faster?

## Observations relevant to the agent design
What would you want an RL agent to remember between frames?
What kinds of objects/shapes appeared across games?
Would a "state change detector" be enough, or do you need semantics?
```

**Success criterion:** The notebook file exists with substantive
answers (not one-liners).

---

### Priority 3 — Real human baselines (60-minute time box)

The current `data/human_baselines.json` uses fallback placeholders. Any
RHAE number computed against these is meaningless.

**Time-box this to 60 minutes.** If you can't find real baselines in
that time, take option B (document and defer).

### Option A — Try to find real baselines (try in order, stop when one works)

#### A.1 — Inspect the SDK source (10 min)

```bash
python -c "import arc_agi, os; print(os.path.dirname(arc_agi.__file__))"
```

Then search that directory for baseline-related code:
```bash
SDK_DIR=$(python -c "import arc_agi, os; print(os.path.dirname(arc_agi.__file__))")
find "$SDK_DIR" -name "*.py" | xargs grep -l -i "baseline\|scorecard\|human" 2>/dev/null
```

Read any matches. Look for method names or URL patterns your
`fetch_baselines.py` didn't try. Common patterns:
- `arc.get_scorecard(game_id)`
- `env.get_human_baseline()`
- A REST endpoint like `api.arcprize.org/scorecard/{game}`

If you find something, update `scripts/fetch_baselines.py` and re-run
`make fetch-baselines`.

#### A.2 — Inspect env/arc attributes (10 min)

```bash
python -c "
import arc_agi
arc = arc_agi.Arcade()
print('Arcade attrs:', sorted([a for a in dir(arc) if not a.startswith('_')]))
env = arc.make('ft09')
print('Env attrs:', sorted([a for a in dir(env) if not a.startswith('_')]))
"
```

Look for names containing: card, score, human, baseline, metrics,
stats, leaderboard, metadata, info.

#### A.3 — Check the ARC-AGI-3 docs (10 min)

Browse https://docs.arcprize.org — search for "baseline", "scorecard",
"RHAE". The docs may document a REST API you can call directly
instead of via the SDK.

#### A.4 — Manually copy from the website (20 min)

If the SDK doesn't expose baselines, read them directly from
https://three.arcprize.org:

1. Go to the game page for `ls20`
2. Find the human performance data (second-best human action count per
   level)
3. Write these numbers into `data/human_baselines.json` manually
4. Change the `"source"` field to `"manual"` so we know where they came
   from
5. Repeat for `ft09`

Example of manually-edited entry:
```json
"ls20": {
    "levels": [7, 11, 14, 19, 24],
    "source": "manual",
    "retrieved_from": "https://three.arcprize.org/ls20",
    "retrieved_at": "2026-04-XX"
}
```

### Option B — Accept and document (if 60 min ran out)

Edit `EXPERIMENTS.md`:

```markdown
### Known issue: human baselines (deferred to Week 5)

**Status:** PLACEHOLDER values as of Week 1.

**Attempts made:**
- fetch_baselines.py: tried 3 SDK access patterns, none exposed baselines
- SDK source inspection: [what you found or didn't]
- Manual website lookup: [attempted/blocked/not tried]

**Impact:** Any RHAE number before this is fixed is meaningless for
external comparison. Still useful for *relative* progress tracking
across our own iterations, since the same placeholder is applied
consistently.

**Must fix before:** Week 5 first evaluation, or any competition submission.

**Next attempt:** Check ARC Prize Discord / forums / GitHub issues for
the correct API path.
```

**Success criterion:** Either real baselines in the JSON, OR a clear
documented-issue entry in `EXPERIMENTS.md`. Don't leave it ambiguous.

---

### Priority 4 — Verify data exploration (30 minutes)

Open `notebooks/01_sdk_exploration.ipynb` and confirm:

#### 4.1 Does it actually visualise frames?

Run all cells. You should see at least:
- A plot of 12-16 sampled frames as RGB images
- Action distribution histogram
- State-change rate

If the notebook only has `ReplayBuffer.load()` and `buf.stats()` without
any actual frame rendering, add these cells:

```python
import matplotlib.pyplot as plt
import numpy as np
from arcwm.env.replay_buffer import ReplayBuffer

buf = ReplayBuffer.load("./data/replay")
batch = buf.sample(16)

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    frame = np.transpose(batch["frame"][i], (1, 2, 0))
    ax.imshow(frame)
    ax.set_title(f"a={batch['action'][i]} r={batch['reward'][i]:.2f}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("notebooks/sample_frames.png", dpi=80)
plt.show()
```

Save the output image (commit `sample_frames.png` to the repo — it's
small, < 100KB, and gives future-you a reference point).

#### 4.2 Add terminal-frame visualisation

Look at frames just before GAME_OVER — these are the most informative:

```python
import numpy as np

terminals = buf.fields["terminal"][:len(buf)]
terminal_indices = np.where(terminals)[0]
print(f"Terminal transitions in buffer: {len(terminal_indices)}")

if len(terminal_indices) > 0:
    # Sample 8 terminal transitions, show the frame that ended the episode
    sample_idx = np.random.choice(terminal_indices, size=min(8, len(terminal_indices)), replace=False)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, idx in zip(axes.flat, sample_idx):
        # Show the NEXT_FRAME (the one after the killing action)
        frame = np.transpose(np.array(buf.next_frames[idx]), (1, 2, 0))
        ax.imshow(frame)
        ax.set_title(f"action={buf.fields['action'][idx]} "
                     f"won={bool(buf.fields['win'][idx])}")
        ax.axis("off")
    plt.suptitle("Frames right after terminal action")
    plt.tight_layout()
    plt.savefig("notebooks/terminal_frames.png", dpi=80)
    plt.show()
```

#### 4.3 Write observations in EXPERIMENTS.md

After viewing the frames, add to `EXPERIMENTS.md` under Week 1:

```markdown
### Data observations (from notebook 01)

**Visual content:**
- [what kinds of shapes/colours dominate]
- [is there a consistent object size?]
- [how different do ls20 and ft09 look?]

**Action effects:**
- Action distribution: [roughly uniform? any bias?]
- State-change rate: X%
- [which actions seem to cause visible change most often?]

**Terminal states:**
- [what does a GAME_OVER frame look like? anything distinctive?]
- [do wins have a distinctive visual signature?]

**Design implications for Week 2:**
- [what does the encoder need to preserve?]
- [is a VAE reconstruction loss likely to work, or do we need
   segmentation-aware losses?]
```

**Success criterion:** The notebook has frame visualisations saved as
PNG, and EXPERIMENTS.md has concrete observations.

---

## End-of-week verification

Before closing Week 1 and opening Week 2, run through this checklist:

- [ ] `make test` → 19 tests pass
- [ ] `make buffer-stats` → both games ≥ 8000 transitions each
- [ ] `notebooks/00_playing_notes.md` exists with substantive answers
- [ ] `data/human_baselines.json` has real values OR `EXPERIMENTS.md`
      documents the issue clearly
- [ ] `notebooks/01_sdk_exploration.ipynb` has visualisations
- [ ] `notebooks/sample_frames.png` and `notebooks/terminal_frames.png`
      committed to repo
- [ ] `EXPERIMENTS.md` has: data rebalance note, data observations,
      baseline status note
- [ ] All changes pushed to GitHub

When all boxes checked, reply with:
- Output of `make buffer-stats`
- Link to the updated repo
- One paragraph summary of what surprised you most in the data or
  gameplay

Then we start Week 2 — world model design.

---

## Hard rules

1. **Don't start Week 2 until Priority 1 and 2 are done.**
   Playing the game and having balanced data are prerequisites, not
   optional extras.
2. **Priority 3 (baselines) is time-boxed at 60 min.** Don't spend all
   of Sunday on it. Document and move on if stuck.
3. **Priority 4 is non-negotiable for quality** but shouldn't take
   more than 30 min if data is already collected.
4. **Commit after each priority completes.** Small commits, clear
   messages.
5. **If Priority 1 collection fails or stalls**, flag it — there might
   be an SDK issue that also affects Week 2 training.

---

## Rough time budget

| Priority | Time        | Can run unattended? |
|----------|-------------|---------------------|
| 1 — Data rebalance | 60 min | Yes, after kickoff |
| 2 — Playing notes  | 30 min | No — active play |
| 3 — Baselines      | 60 min cap | Partial |
| 4 — Exploration    | 30 min | No — need to inspect output |
| **Total**          | **~3 hours** |                     |

If you can overlap Priority 1 (running in background) with Priority 2
(playing the game), you save ~1 hour.
