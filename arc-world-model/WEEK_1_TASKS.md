# WEEK 1 — Infrastructure & Data Collection

**Goal:** A working repo, real human baselines, and 10K+ random transitions
saved to disk. No training yet.

**Estimated time:** 15–20 hours
**Compute budget:** ~5 GPU-hours (data collection is CPU-bound; GPU only if
you want to prototype in a notebook)

---

## Before starting — play the game

**Required.** Spend 30 minutes at [three.arcprize.org](https://three.arcprize.org).
Play 2–3 games. Pay attention to:

- How do you figure out what the actions do?
- What visual cues tell you you're making progress?
- How do you recognise a "win"?

Write a paragraph in `notebooks/00_playing_notes.md` about what you
noticed. This informs every design decision in the weeks ahead.

---

## Task 1 — Repo setup (1–2 hours)

### 1.1 Clone and install

```bash
# Create the repo on GitHub first (empty, no README)
# Then locally:
cd ~/projects  # or wherever
git init arc-world-model
cd arc-world-model

# Copy all files from the starter pack into this directory
# (the zip we provided)

git add .
git commit -m "Initial commit: scaffold from arcwm starter pack"
git branch -M main
git remote add origin git@github.com:asiaat/arc-world-model.git
git push -u origin main
```

### 1.2 Install

```bash
# Create venv with Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev extras
pip install --upgrade pip
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env and fill in ARC_API_KEY
```

### 1.3 Verify

```bash
make test
```

**Success criterion:** 17 tests pass.

### 1.4 Verify SDK works

```bash
python -c "
import arc_agi
from arcengine import GameAction, GameState
arc = arc_agi.Arcade()
env = arc.make('ft09')
obs = env.reset()
print('SDK OK. Observation type:', type(obs).__name__)
print('Has frame:', hasattr(obs, 'frame'))
"
```

**Success criterion:** Prints `SDK OK.` without error.

**If it fails:** the ARC_API_KEY is wrong, or `arc-agi` package is not
installed. Fix before proceeding.

---

## Task 2 — Fetch human baselines (30 minutes)

```bash
make fetch-baselines
```

This writes `data/human_baselines.json`.

Open the file and inspect. You will see one of two things:

### Case A — SDK provided real baselines

The file shows `"source": "sdk"` for each game. Good — these are the
real numbers used for RHAE scoring.

### Case B — SDK did not provide baselines

The file shows `"source": "fallback"` for some or all games. This means
the SDK version does not expose baselines through any of the access
patterns we tried. You need to find the real numbers elsewhere.

**Where to look:**

1. **The scorecard endpoint** — some SDK versions expose
   `arc.scorecard()` or similar. Check the SDK source:
   ```bash
   python -c "import arc_agi; print(arc_agi.__file__)"
   # Then inspect the installed package, look for 'scorecard' or 'baseline'
   ```
2. **The ARC-AGI-3 website** — the leaderboard page shows human baselines
   per game. Manually copy the numbers into
   `data/human_baselines.json`.
3. **The arcprize.org API** — there may be a REST endpoint. Check the SDK
   docs.

If you cannot find real baselines after 30 minutes of looking, flag this
and proceed with placeholders. We will fix them before evaluating in
Week 5.

**Success criterion:** `data/human_baselines.json` exists with entries
for both games in `TRAIN_GAMES`.

---

## Task 3 — Collect 10K transitions (2–4 hours, mostly unattended)

```bash
make collect-data
```

This runs random-policy episodes on both games until 10K transitions are
stored. Expected behaviour:

- Each episode is 50–500 steps before GAME_OVER (no wins expected)
- Progress printed every episode
- Buffer flushed to disk every 30 seconds (safe to Ctrl+C)
- Rate: 50–200 transitions/second depending on SDK latency

### Monitor the run

In a second terminal:

```bash
watch -n 5 'python -m arcwm.env.replay_buffer stats --path ./data/replay'
```

### If it crashes or stalls

- **SDK timeout** — the SDK sometimes hangs on env creation. Ctrl+C and
  re-run, the buffer resumes from where it was.
- **Disk full** — 10K transitions ≈ 250 MB. 100K ≈ 2.5 GB. Check with `df -h`.
- **Slow rate (<10/s)** — SDK connection may be flaky. Try again later;
  if persistent, check ARC-AGI-3 status page.

### After completion

```bash
make buffer-stats
```

**Expected output:**

```json
{
  "size": 10000,
  "capacity": 20000,
  "reward": {
    "min": -1.0,
    "max": 0.05,
    "mean": 0.002,
    "nonzero_frac": 0.15
  },
  "terminals": 40,
  "wins": 0,
  "per_game": {
    "ls20": { "transitions": 5000, "terminals": 20, "wins": 0 },
    "ft09": { "transitions": 5000, "terminals": 20, "wins": 0 }
  }
}
```

**Success criteria:**
- `size >= 10000`
- Both games represented with non-trivial counts
- `terminals > 0` (episodes actually ended in GAME_OVER)
- `wins = 0` is expected; random can't win (we proved this in the old repo)

---

## Task 4 — Explore the data (2–3 hours)

Open `notebooks/01_sdk_exploration.ipynb` (create it) and do the
following:

### 4.1 Load the buffer and inspect

```python
from arcwm.env.replay_buffer import ReplayBuffer
buf = ReplayBuffer.load("./data/replay")
print(f"Size: {len(buf)}")
print(buf.stats())
```

### 4.2 Visualise frames

```python
import matplotlib.pyplot as plt
import numpy as np

batch = buf.sample(16)
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    frame = np.transpose(batch["frame"][i], (1, 2, 0))
    ax.imshow(frame)
    ax.set_title(f"a={batch['action'][i]} r={batch['reward'][i]:.2f}")
    ax.axis("off")
plt.tight_layout()
plt.show()
```

Look at 50–100 frames. **Write down observations** in the notebook:

- How much visual variety is there?
- Are there recognisable objects that appear across games?
- Do certain actions obviously change certain things?
- What does a frame right before GAME_OVER look like?

This is not busywork — these observations shape what the encoder should
capture.

### 4.3 Action histogram

```python
import collections
actions = buf.fields["action"][:len(buf)]
counts = collections.Counter(actions.tolist())
print("Action distribution:", dict(sorted(counts.items())))
```

**Expected:** roughly uniform (~670 per action for 10K/15 actions). If
wildly skewed, the collection script has a bug.

### 4.4 State-change rate

```python
changed = (batch["reward"] > 0).mean()
print(f"Frame change rate: {changed:.1%}")
```

**Expected:** 10–30%. Most actions do nothing, which is consistent with
ARC-AGI-3 design.

**Success criterion:** You have looked at the data and have a sense of
what the agent is working with. This is your intuition for the rest of
the project.

---

## Task 5 — Update `program.md` equivalent (30 minutes)

Create `EXPERIMENTS.md` at the repo root (the equivalent of `program.md`
in the old repo). Add:

```markdown
# arcwm — Experiments log

## Week 1 — Data collection

**Dates:** 2026-04-XX to 2026-04-XX

### Setup
- Repo created, 17 tests pass
- SDK verified: obs.frame is list of 2D color-index arrays, 0-15 range
- Human baselines: [sdk / fallback] for each game
- Replay buffer: 10K transitions on games [ls20, ft09]

### Observations
[Your notes from Task 4.2 — what the frames look like, action patterns,
state-change rate. Be concrete.]

### Questions for Week 2
- [List any questions the data raised that will inform world model design]
```

**Success criterion:** a written record. Future weeks will extend this
file.

---

## Hard rules for this week

1. **No model training.** Even if you finish data collection early and
   want to try a CNN, don't. Week 2 starts Monday. Use extra time to
   understand the data better.
2. **Commit often.** After each task, `git commit` with a clear message.
3. **Test before committing.** `make test` must pass on every commit.
4. **Keep the dev loop fast.** No PRs to yourself, no elaborate
   branching. `main` only. This is a 10-week solo project; overhead
   kills it.
5. **Log what surprises you.** If something in the data looks weird,
   write it down immediately. Surprises in Week 1 explain failures in
   Weeks 3-5.

---

## End-of-week checklist

By Sunday:

- [ ] Played 2-3 games at three.arcprize.org, wrote notes
- [ ] Repo pushed to github.com/asiaat/arc-world-model
- [ ] `make test` passes (17 tests)
- [ ] `make fetch-baselines` ran, `data/human_baselines.json` exists
- [ ] `make collect-data` completed, 10K+ transitions in buffer
- [ ] Opened the data in a notebook, viewed 50+ frames
- [ ] `EXPERIMENTS.md` has the Week 1 entry with observations
- [ ] Initial thoughts on encoder design written down

Then report back and we start Week 2 — world model v1.

---

## What's next (preview)

**Week 2:** We build the world model. Specifically:
- VAE encoder/decoder that compresses frames to 64-d latents
- Transition model `p(z_{t+1} | z_t, a_t)` trained on the buffer
- Success criterion: open-loop prediction of 5 steps into the future
  produces frames that visually resemble actual gameplay

The quality of Week 2 depends entirely on Week 1 data. Don't skimp.
