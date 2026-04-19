"""Tests specifically for the replay buffer resume-after-load behaviour.

Separate file because this tests a specific bug: when a buffer is loaded
from disk, the fields arrays must be expanded back to full capacity so
that further `add()` calls work.
"""

import numpy as np

from arcwm.env.arc_wrapper import Transition
from arcwm.env.replay_buffer import ReplayBuffer


def _make_transition(action=0, reward=0.0, terminal=False, win=False, game="test"):
    return Transition(
        frame=np.full((3, 64, 64), action, dtype=np.uint8),
        action=action,
        reward=reward,
        next_frame=np.full((3, 64, 64), action + 1, dtype=np.uint8),
        terminal=terminal,
        win=win,
        game_id=game,
        step_idx=0,
    )


class TestResume:
    def test_can_add_after_load(self, tmp_path):
        """Loading a buffer and adding more must not crash."""
        path = tmp_path / "buf"
        buf = ReplayBuffer.create(path, capacity=100)
        for i in range(10):
            buf.add(_make_transition(action=i))
        buf.flush()

        # Reload and keep adding
        buf2 = ReplayBuffer.load(path)
        assert len(buf2) == 10
        for i in range(10, 20):
            buf2.add(_make_transition(action=i))
        assert len(buf2) == 20
        buf2.flush()

        # Reload again — data from both sessions should be there
        buf3 = ReplayBuffer.load(path)
        assert len(buf3) == 20
        actions_in_buffer = buf3.fields["action"][:20]
        assert set(actions_in_buffer.tolist()) == set(range(20))

    def test_game_id_mapping_preserved_across_load(self, tmp_path):
        path = tmp_path / "buf"
        buf = ReplayBuffer.create(path, capacity=50)
        buf.add(_make_transition(game="alpha"))
        buf.add(_make_transition(game="beta"))
        buf.flush()

        buf2 = ReplayBuffer.load(path)
        # Adding a transition with an existing game_id should reuse its index
        buf2.add(_make_transition(game="alpha"))
        # Adding a new game should get a new index
        buf2.add(_make_transition(game="gamma"))
        buf2.flush()

        stats = buf2.stats()
        assert stats["size"] == 4
        assert "alpha" in stats["per_game"]
        assert "beta" in stats["per_game"]
        assert "gamma" in stats["per_game"]
        assert stats["per_game"]["alpha"]["transitions"] == 2
        assert stats["per_game"]["beta"]["transitions"] == 1
        assert stats["per_game"]["gamma"]["transitions"] == 1
