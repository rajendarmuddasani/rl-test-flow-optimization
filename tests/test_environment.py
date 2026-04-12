"""Tests for TestFlowEnv gymnasium environment."""

import numpy as np
import pytest
from src.environment import TestFlowEnv, build_default_catalog, DEFECT_CATEGORIES, CATEGORY_GROUPS


@pytest.fixture
def env():
    return TestFlowEnv(n_tests=10)


class TestTestFlowEnv:
    def test_reset_shape(self, env):
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)

    def test_action_space(self, env):
        assert env.action_space.n == env.n_tests + 1

    def test_run_test_returns_negative_reward(self, env):
        env.reset(seed=42)
        obs, reward, term, trunc, info = env.step(0)
        assert reward < 0
        assert not term

    def test_stop_action_ends_episode(self, env):
        env.reset(seed=42)
        stop_action = env.n_tests
        obs, reward, term, trunc, info = env.step(stop_action)
        assert term
        assert "correct" in info

    def test_duplicate_test_penalty(self, env):
        env.reset(seed=42)
        env.step(0)
        obs, reward, _, _, info = env.step(0)
        assert reward == -1.0
        assert info.get("error") == "duplicate_test"

    def test_budget_exceeded_forces_stop(self):
        tight_env = TestFlowEnv(n_tests=10, cost_budget=2.0)
        tight_env.reset(seed=42)
        # Find a test that exceeds budget
        for i in range(tight_env.n_tests):
            if tight_env.tests[tight_env.test_names[i]]["cost"] > 2.0:
                obs, reward, term, trunc, info = tight_env.step(i)
                assert term
                break

    def test_action_masks(self, env):
        env.reset(seed=42)
        masks = env.action_masks()
        assert masks.shape == (env.n_tests + 1,)
        assert masks.dtype == bool
        assert masks[env.n_tests]  # STOP always valid
        assert masks[:env.n_tests].all()  # all tests valid initially

    def test_action_masks_update_after_step(self, env):
        env.reset(seed=42)
        env.step(0)
        masks = env.action_masks()
        assert not masks[0]  # already run
        assert masks[env.n_tests]  # STOP still valid

    def test_build_default_catalog(self):
        catalog = build_default_catalog(100, seed=42)
        assert len(catalog) == 100
        for name, info in catalog.items():
            assert "cost" in info
            assert "time" in info
            assert "group" in info
            assert "defect_coverage" in info

    def test_defect_categories(self):
        assert len(DEFECT_CATEGORIES) == 14
        assert len(CATEGORY_GROUPS) == 5

    def test_scalable_env(self):
        for n in [10, 50, 200]:
            env = TestFlowEnv(n_tests=n)
            obs, _ = env.reset(seed=42)
            assert obs.shape[0] == n * 5 + 3
            assert env.action_space.n == n + 1

    def test_full_episode_metrics(self, env):
        env.reset(seed=42)
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
            if steps > 20:
                break  # safety
        assert steps > 0

    def test_defect_rate(self):
        """Check that defect_rate parameter controls chip generation."""
        env = TestFlowEnv(defect_rate=1.0)
        defects = 0
        for _ in range(50):
            env.reset()
            if env._defect is not None:
                defects += 1
        assert defects == 50

    def test_good_chip_rate(self):
        env = TestFlowEnv(defect_rate=0.0)
        for _ in range(20):
            env.reset()
            assert env._defect is None
