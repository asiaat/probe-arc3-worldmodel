# arcwm — ARC-AGI-3 World Model Agent

A model-based reinforcement learning agent for the
[ARC Prize 2026](https://arcprize.org/competitions/2026) ARC-AGI-3 competition.

> **Status:** Active development. Targeting Milestone #1 (June 30, 2026).

## Approach

Instead of model-free RL (PPO + curiosity), this agent learns a **world model**
— a compact neural representation of how the game environment evolves — and
then does **search in imagination** to find winning action sequences.

```
Frame (64×64 RGB) → Encoder → latent z_t
                               │
                               ▼
                        RSSM transition model
                               │
                    p(z_{t+1} | z_t, a_t)
                               │
                     MCTS over imagined rollouts
                               │
                          best action a_t
```

Components:

- **Encoder/Decoder** — CNN VAE that maps frames to/from a 64-d latent
- **RSSM** — Recurrent state-space model for latent dynamics
- **Ensemble** — 5 transition models for epistemic uncertainty → intrinsic reward
- **MCTS planner** — Monte Carlo Tree Search over imagined latent rollouts
- **Distilled policy** — Fast reactive policy that imitates MCTS for eval speed

## Installation

Requires Python 3.12+ (ARC-AGI-3 SDK dependency).

```bash
git clone https://github.com/asiaat/arc-world-model.git
cd arc-world-model

# Create venv
python3.12 -m venv .venv
source .venv/bin/activate

# Install package
pip install -e .

# Configure
cp .env.example .env
# Edit .env — add your ARC_API_KEY

# Verify setup
make test
```

## Quick start

```bash
# Week 1 workflow
make fetch-baselines     # Get real human baselines from SDK
make collect-data        # Collect 10K random transitions
make buffer-stats        # Show buffer statistics

# Later weeks
make train-wm            # Train world model
make train-agent         # Full model-based RL training
make eval                # Evaluate on held-out games
```

## Project structure

```
arcwm/
├── env/                    # SDK wrapper, replay buffer
├── models/                 # Encoder, RSSM, heads, ensemble
├── planning/               # MCTS, CEM planners
├── training/               # World model + policy training loops
├── agent.py                # Main agent class
└── eval.py                 # RHAE evaluation

scripts/
├── collect_data.py         # Data collection (Week 1)
├── fetch_baselines.py      # Human baselines (Week 1)
├── train_world_model.py    # Week 2+
├── train_agent.py          # Week 5+
└── kaggle_submit.py        # Week 9

configs/
├── default.yaml            # Production config
└── small.yaml              # Fast iteration config
```

## Timeline

| Week | Deliverable |
|------|-------------|
| 1    | Infrastructure, data collection, real baselines |
| 2    | World model v1 — VAE + transition, 5-step predictions |
| 3    | CEM planner in latent space |
| 4    | Ensemble + intrinsic reward from disagreement |
| 5    | First wins (target) |
| 6    | MCTS upgrade |
| 7    | Policy distillation |
| 8    | 48-hour training run |
| 9    | Kaggle packaging |
| 10   | Submission (June 30 deadline) |

## License

MIT-0 (required for ARC Prize eligibility). See [LICENSE](LICENSE).

## Acknowledgements

- [ARC Prize Foundation](https://arcprize.org) for the benchmark
- Architecture inspired by [Dreamer V3](https://arxiv.org/abs/2301.04104)
  and [MuZero](https://arxiv.org/abs/1911.08265)
