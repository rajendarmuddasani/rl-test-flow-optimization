"""
Gymnasium environment for semiconductor test flow optimization.

Supports 10–1000 tests with action masking. The agent selects which test
to run next; the episode ends on STOP or budget exhaustion.

Key design choices for large action spaces:
  - action_masks() method for SB3 MaskablePPO compatibility
  - Observation factored into per-test features + global state
  - Configurable cost/time budgets and defect taxonomy
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)


# ── Default defect categories ──────────────────────────────────────────────
DEFECT_CATEGORIES = [
    "voltage_droop", "current_leak", "esd_damage",
    "setup_violation", "hold_violation", "clock_jitter", "frequency_fail",
    "power_thermal", "electromigration",
    "logic_fail", "memory_fail", "io_fail", "scan_fail",
    "analog_drift",
]

CATEGORY_GROUPS = {
    "electrical": ["voltage_droop", "current_leak", "esd_damage"],
    "timing": ["setup_violation", "hold_violation", "clock_jitter", "frequency_fail"],
    "thermal": ["power_thermal", "electromigration"],
    "functional": ["logic_fail", "memory_fail", "io_fail", "scan_fail"],
    "analog": ["analog_drift"],
}

_DEFECT_TO_GROUP: dict[str, str] = {}
for g, members in CATEGORY_GROUPS.items():
    for d in members:
        _DEFECT_TO_GROUP[d] = g


def load_test_catalog(path: str | Path) -> dict:
    """Load test catalog from JSON file."""
    with open(path) as f:
        return json.load(f)


def build_default_catalog(n_tests: int = 10, seed: int = 42) -> dict:
    """Build a synthetic test catalog with n_tests entries."""
    rng = np.random.RandomState(seed)
    groups = list(CATEGORY_GROUPS.keys())
    catalog: dict = {}
    for i in range(n_tests):
        group = groups[i % len(groups)]
        catalog[f"TEST_{i:04d}"] = {
            "cost": round(float(rng.uniform(1, 20)), 2),
            "time": round(float(rng.uniform(2, 60)), 2),
            "group": group,
            "defect_coverage": round(float(rng.uniform(0.05, 0.40)), 3),
        }
    return catalog


class TestFlowEnv(gym.Env):
    """Adaptive test flow selection environment.

    Observation (Box):
        Per-test block (5 features × n_tests):
            [already_run, result, norm_cost, norm_time, coverage]
        Global (3 features):
            [budget_remaining_frac, time_remaining_frac, tests_run_frac]

    Action (Discrete n_tests + 1):
        0..n_tests-1 = run test i
        n_tests      = STOP

    action_masks() returns a boolean array for SB3 MaskablePPO.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        test_config: dict | None = None,
        n_tests: int = 10,
        cost_budget: float = 30.0,
        time_budget: float = 60.0,
        defect_rate: float = 0.70,
        seed: int = 42,
        max_steps_per_episode: int | None = None,
    ):
        super().__init__()
        self._max_steps = max_steps_per_episode if max_steps_per_episode is not None else n_tests * 2 + 10
        if test_config is not None:
            self.tests = test_config
        else:
            self.tests = build_default_catalog(n_tests, seed)
        self.test_names = list(self.tests.keys())
        self.n_tests = len(self.test_names)
        self.cost_budget = cost_budget
        self.time_budget = time_budget
        self.defect_rate = defect_rate

        self.action_space = spaces.Discrete(self.n_tests + 1)
        obs_dim = self.n_tests * 5 + 3
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32,
        )

        self._max_cost = max(t["cost"] for t in self.tests.values())
        self._max_time = max(t["time"] for t in self.tests.values())

        # Pre-compute group for each test
        self._test_groups = [self.tests[n].get("group", "functional") for n in self.test_names]

        # ── Pre-computed numpy arrays for vectorised hot paths ─────────────
        # These are read-only across all steps — built once in __init__.
        costs = np.array([self.tests[n]["cost"] for n in self.test_names], dtype=np.float32)
        times = np.array([self.tests[n]["time"] for n in self.test_names], dtype=np.float32)
        covs  = np.array([self.tests[n]["defect_coverage"] for n in self.test_names], dtype=np.float32)
        self._raw_costs   = costs                          # for budget check
        self._norm_costs  = costs / self._max_cost         # for obs
        self._norm_times  = times / self._max_time         # for obs
        self._coverages   = covs                           # for obs
        self._stop_true   = np.array([True], dtype=bool)   # STOP always valid

    # ── Reset / Step ───────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._run_mask = np.zeros(self.n_tests, dtype=np.float32)
        self._results = np.full(self.n_tests, -1.0, dtype=np.float32)
        self._cost_spent = 0.0
        self._time_spent = 0.0
        self._done = False
        self._step_count = 0

        # Sample chip
        if self.np_random.random() < self.defect_rate:
            self._defect = self.np_random.choice(DEFECT_CATEGORIES)
        else:
            self._defect = None

        return self._obs(), {}

    def step(self, action: int):
        if self._done:
            return self._obs(), 0.0, True, False, {}

        self._step_count += 1

        # Guard: force-terminate if episode exceeds max_steps (prevents infinite
        # loops when a deterministic policy repeatedly selects duplicate actions,
        # since duplicates don't change state → same obs → same prediction).
        if self._step_count >= self._max_steps:
            return self._evaluate()

        # STOP
        if action == self.n_tests:
            return self._evaluate()

        if action < 0 or action >= self.n_tests:
            return self._obs(), -1.0, False, False, {"error": "invalid_action"}

        # Duplicate test
        if self._run_mask[action] == 1.0:
            return self._obs(), -1.0, False, False, {"error": "duplicate_test"}

        test_info = self.tests[self.test_names[action]]
        cost = test_info["cost"]

        # Budget exceeded → forced stop
        if self._cost_spent + cost > self.cost_budget:
            return self._evaluate()

        self._run_mask[action] = 1.0
        self._cost_spent += cost
        self._time_spent += test_info["time"]

        # Simulate test result
        result = self._simulate_result(action)
        self._results[action] = result

        reward = -cost / self._max_cost - 0.1 * test_info["time"] / self._max_time
        return self._obs(), reward, False, False, {
            "test": self.test_names[action], "result": result,
        }

    # ── Action Masking (for MaskablePPO) ───────────────────────────────────

    def action_masks(self) -> np.ndarray:
        """Return boolean mask of valid actions — fully vectorised (no Python loop)."""
        not_run   = self._run_mask == 0.0                            # shape (n_tests,)
        affordable = self._raw_costs <= (self.cost_budget - self._cost_spent)
        return np.concatenate([not_run & affordable, self._stop_true])

    # ── Internal ───────────────────────────────────────────────────────────

    def _simulate_result(self, action: int) -> float:
        """Simulate pass/fail for a test on the current chip."""
        if self._defect is None:
            return 1.0 if self.np_random.random() > 0.05 else 0.0

        defect_group = _DEFECT_TO_GROUP.get(self._defect, "")
        test_group = self._test_groups[action]
        if test_group == defect_group:
            return 0.0 if self.np_random.random() > 0.20 else 1.0
        else:
            return 1.0 if self.np_random.random() > 0.30 else 0.0

    def _evaluate(self):
        """End episode: classify chip, compute final reward."""
        self._done = True
        run_indices = self._run_mask == 1.0
        any_fail = bool(np.any(self._results[run_indices] == 0.0)) if run_indices.any() else False

        if self._defect is not None:
            reward = 10.0 if any_fail else -20.0
            correct = bool(any_fail)
        else:
            reward = 5.0 if not any_fail else -5.0
            correct = not any_fail

        info = {
            "correct": correct,
            "defect": self._defect,
            "cost": self._cost_spent,
            "time": self._time_spent,
            "tests_run": int(self._run_mask.sum()),
        }
        return self._obs(), reward, True, False, info

    def _obs(self) -> np.ndarray:
        # Vectorised — no Python loop, uses pre-built numpy arrays.
        per_test = np.empty(self.n_tests * 5, dtype=np.float32)
        per_test[0::5] = self._run_mask       # already_run
        per_test[1::5] = self._results        # test result
        per_test[2::5] = self._norm_costs     # normalised cost
        per_test[3::5] = self._norm_times     # normalised time
        per_test[4::5] = self._coverages      # defect coverage

        global_feats = np.array([
            1.0 - self._cost_spent / self.cost_budget,
            1.0 - self._time_spent / self.time_budget,
            self._run_mask.sum() / self.n_tests,
        ], dtype=np.float32)

        return np.concatenate([per_test, global_feats])
