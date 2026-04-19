"""Tests for the env wrapper and replay buffer."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from arcwm.env.arc_wrapper import (
    COLOR_MAP, NUM_ACTIONS, Transition, action_idx_to_game_action,
    frame_to_rgb, parse_state, TerminalState,
)
from arcwm.env.replay_buffer import ReplayBuffer


class TestFrameConversion:
    def test_none_input_returns_zeros(self):
        out = frame_to_rgb(None)
        assert out.shape == (3, 64, 64)
        assert out.dtype == np.uint8
        assert out.sum() == 0

    def test_single_layer_2d(self):
        layer = np.zeros((64, 64), dtype=np.int32)
        layer[10:20, 10:20] = 1  # a 10x10 blue block
        out = frame_to_rgb(layer)
        assert out.shape == (3, 64, 64)
        # Blue index 1 → (0, 116, 217)
        assert out[0, 15, 15] == 0
        assert out[1, 15, 15] == 116
        assert out[2, 15, 15] == 217

    def test_list_of_layers_composites_correctly(self):
        base = np.zeros((64, 64), dtype=np.int32)
        base[:, :] = 1  # all blue background
        overlay = np.zeros((64, 64), dtype=np.int32)
        overlay[30:40, 30:40] = 2  # red square overlay
        out = frame_to_rgb([base, overlay])
        # Pixel at (10, 10) should be blue (only base)
        assert tuple(out[:, 10, 10]) == (0, 116, 217)
        # Pixel at (35, 35) should be red (overlay wins)
        assert tuple(out[:, 35, 35]) == (255, 65, 54)

    def test_resize_to_64(self):
        small = np.zeros((32, 32), dtype=np.int32)
        small[:, :] = 1
        out = frame_to_rgb(small)
        assert out.shape == (3, 64, 64)

    def test_empty_list_returns_zeros(self):
        out = frame_to_rgb([])
        assert out.shape == (3, 64, 64)
        assert out.sum() == 0

    def test_invalid_indices_clipped(self):
        layer = np.zeros((64, 64), dtype=np.int32)
        layer[0, 0] = 99  # out of palette range
        out = frame_to_rgb(layer)
        # Should clip to 15 (light gray)
        assert tuple(out[:, 0, 0]) == (200, 200, 200)


class TestActionMapping:
    def test_action_range(self):
        assert NUM_ACTIONS == 15

    def test_action_indices_return_tuple(self):
        for idx in range(NUM_ACTIONS):
            result = action_idx_to_game_action(idx)
            assert isinstance(result, tuple)
            assert len(result) == 2


class TestColorMap:
    def test_color_map_shape(self):
        assert COLOR_MAP.shape == (16, 3)
        assert COLOR_MAP.dtype == np.uint8

    def test_color_map_has_black_and_white(self):
        assert tuple(COLOR_MAP[0]) == (0, 0, 0)  # black
        assert tuple(COLOR_MAP[12]) == (255, 255, 255)  # white


class TestTerminalState:
    def test_parse_none_is_playing(self):
        assert parse_state(None) == TerminalState.PLAYING


class TestReplayBuffer:
    def _make_transition(self, action=0, reward=0.0, terminal=False, win=False, game="test"):
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

    def test_create_and_add(self, tmp_path):
        buf = ReplayBuffer.create(tmp_path / "buf", capacity=10)
        assert len(buf) == 0
        buf.add(self._make_transition(action=1, reward=0.5))
        assert len(buf) == 1
        buf.flush()

    def test_persist_and_reload(self, tmp_path):
        path = tmp_path / "buf"
        buf = ReplayBuffer.create(path, capacity=10)
        for i in range(5):
            buf.add(self._make_transition(action=i, reward=float(i)))
        buf.flush()

        # Reload from disk
        buf2 = ReplayBuffer.load(path)
        assert len(buf2) == 5
        batch = buf2.sample(batch_size=3)
        assert batch["frame"].shape == (3, 3, 64, 64)
        assert batch["action"].shape == (3,)

    def test_capacity_enforced(self, tmp_path):
        buf = ReplayBuffer.create(tmp_path / "buf", capacity=3)
        for _ in range(3):
            buf.add(self._make_transition())
        with pytest.raises(RuntimeError, match="full"):
            buf.add(self._make_transition())

    def test_stats(self, tmp_path):
        buf = ReplayBuffer.create(tmp_path / "buf", capacity=10)
        buf.add(self._make_transition(reward=1.0, terminal=True, win=True, game="a"))
        buf.add(self._make_transition(reward=0.0, game="a"))
        buf.add(self._make_transition(reward=-1.0, terminal=True, game="b"))
        stats = buf.stats()
        assert stats["size"] == 3
        assert stats["terminals"] == 2
        assert stats["wins"] == 1
        assert "a" in stats["per_game"]
        assert "b" in stats["per_game"]
        assert stats["per_game"]["a"]["transitions"] == 2

    def test_cannot_create_over_existing(self, tmp_path):
        ReplayBuffer.create(tmp_path / "buf", capacity=5)
        with pytest.raises(FileExistsError):
            ReplayBuffer.create(tmp_path / "buf", capacity=5)

    def test_sample_returns_all_fields(self, tmp_path):
        buf = ReplayBuffer.create(tmp_path / "buf", capacity=5)
        buf.add(self._make_transition())
        batch = buf.sample(batch_size=1)
        expected = {"frame", "next_frame", "action", "reward", "terminal",
                    "win", "step_idx", "game_id_idx"}
        assert set(batch.keys()) == expected
