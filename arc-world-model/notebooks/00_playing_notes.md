# Playing notes — ARC-AGI-3 human experience

**Source:** Synthesis of public player comments, ARC docs, and human-study data.

## Games played
- ARC released 3 preview games, later 25 public demo environments
- 342 human step-by-step replays from 458-person study
- ls20: Agent reasoning, spatial navigation, entity control
- ft09: Elementary logic, boolean state deduction, interaction-based transformations
- vc33: Orchestration, multi-object coordination, fluid routing mechanics

## Figuring out actions
**Probing first, then modeling** is the universal pattern:
1. First instinct: test spatial traversal (ACTION1-4) - directional commands
2. If silence: shift to proximal interaction (ACTION5) or coordinate clicking (ACTION6)
3. Key insight: **null actions** (invalid inputs with no state change) are immediately pruned

ACTION6 (coordinate clicking) has 4,096 possibilities - humans filter via visual salience, reducing to 3-4 actionable targets by clicking only on prominent geometric structures.

## Progress signals
**Core Knowledge priors** drive interpretation:
- Objectness: group adjacent pixels into persistent entities
- Basic Geometry: detect symmetry, containment, spatial relationships
- Goal-Directedness: assume agents/mechanisms act toward purpose
- Numbers: track when object counts change

"Useful" action = coherent, non-random structural alteration. A state change that breaks geometric pattern = mistake.

## Recognising wins
- Level transition is the explicit win marker (no music, text, or score screen)
- WIN state is silent - grid clears and new level loads
- "Aha" moment: sudden cascading structural clarity
- Rule from one level transfers to subsequent levels (semantic memory)

## Estimated action efficiency
- Whole preview set: ~500-600 actions
- Some humans needed >600 actions (exploration overhead)
- Repeat play = near-optimal execution (rule internalized)

**Scoring formula:** `(HumanBaseline / AgentActions)^2` - aggressively penalizes inefficiency

## Observations relevant to agent design

### What to remember between frames:
- Which actions caused any change
- Which coordinates were clickable
- What objects persist, where they are
- Current goal hypothesis
- Action sequences that led to progress vs dead ends

### Objects seen:
- Abstract 2D grid-world entities: colored cells, geometric shapes
- Familiar cues: health/turn counters in some games
- Color palette: 16 distinct colors in 64x64 grid

### State-change detector: NOT sufficient
- Human gameplay proves bare state-change detection is inadequate
- A state-change can move you closer OR farther from goal
- Need **semantic world models**: causal rules, entity relationships, teleological goal inference

### Key architectural implications:
1. **Episodic memory**: compress frames to critical state transitions, not raw pixels
2. **Object permanence**: track entities across frames, recognize shapes
3. **Graph-based exploration**: systematic state space, not random walk
4. **Semantic world model over state-change detector**: predict HOW/WHY state changes

**Bottom line:** The benchmark tests exploration, modeling, goal-setting, and planning. Encoder must preserve semantic structure, not just visual fidelity.