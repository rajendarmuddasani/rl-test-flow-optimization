"""FastAPI service for RL test flow optimization inference."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.environment import TestFlowEnv, build_default_catalog

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RL Test Flow Optimizer",
    description="Semiconductor test flow optimization using reinforcement learning",
    version="2.0.0",
)

# ── Global state ───────────────────────────────────────────────────────────

_model = None
_env_config: dict = {}


class OptimizeRequest(BaseModel):
    chip_id: str = Field(..., description="Chip identifier")
    n_tests: int = Field(default=10, ge=1, le=1000, description="Number of tests available")
    cost_budget: float = Field(default=30.0, gt=0, description="Maximum cost budget")
    time_budget: float = Field(default=60.0, gt=0, description="Maximum time budget")
    test_results: dict[str, int] = Field(
        default_factory=dict,
        description="Already-known test results {test_name: 0/1}",
    )


class TestRecommendation(BaseModel):
    test_name: str
    cost: float
    time: float
    group: str
    priority: int


class OptimizeResponse(BaseModel):
    chip_id: str
    recommended_tests: list[TestRecommendation]
    estimated_cost: float
    estimated_time: float
    model_used: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


# ── Endpoints ──────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        version="2.0.0",
    )


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    """Recommend an optimized test sequence for a chip."""
    catalog = build_default_catalog(req.n_tests)
    env = TestFlowEnv(
        test_config=catalog,
        cost_budget=req.cost_budget,
        time_budget=req.time_budget,
    )

    obs, _ = env.reset()

    # Apply any pre-existing results
    for tname, result in req.test_results.items():
        if tname in env.test_names:
            idx = env.test_names.index(tname)
            env._run_mask[idx] = 1.0
            env._results[idx] = float(result)

    # Use trained model if loaded, else greedy heuristic
    if _model is not None:
        model_name = "trained_rl"
        recommendations = _run_model_sequence(env, _model)
    else:
        from src.agent import greedy_coverage_policy

        model_name = "greedy_coverage_heuristic"
        recommendations = _run_heuristic_sequence(env, greedy_coverage_policy)

    total_cost = sum(r.cost for r in recommendations)
    total_time = sum(r.time for r in recommendations)

    return OptimizeResponse(
        chip_id=req.chip_id,
        recommended_tests=recommendations,
        estimated_cost=round(total_cost, 2),
        estimated_time=round(total_time, 2),
        model_used=model_name,
    )


@app.get("/tests")
def list_tests(n_tests: int = 10):
    """List available tests in the catalog."""
    catalog = build_default_catalog(n_tests)
    return {"n_tests": len(catalog), "tests": catalog}


@app.post("/load-model")
def load_model(model_path: str = "outputs/models/maskable_ppo.zip"):
    """Load a trained SB3 model for inference."""
    global _model
    path = Path(model_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
    try:
        from sb3_contrib import MaskablePPO

        _model = MaskablePPO.load(str(path))
        return {"status": "loaded", "path": model_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Helpers ────────────────────────────────────────────────────────────────


def _run_model_sequence(env: TestFlowEnv, model, max_steps: int = 50) -> list[TestRecommendation]:
    """Run trained model to get ordered test recommendations."""
    recs = []
    obs = env._obs()
    for priority in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        if action == env.n_tests:
            break
        if env._run_mask[action] == 1.0:
            continue
        tname = env.test_names[action]
        t = env.tests[tname]
        recs.append(TestRecommendation(
            test_name=tname,
            cost=t["cost"],
            time=t["time"],
            group=t["group"],
            priority=priority + 1,
        ))
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return recs


def _run_heuristic_sequence(env: TestFlowEnv, policy_fn, max_steps: int = 50) -> list[TestRecommendation]:
    """Run heuristic policy to get ordered test recommendations."""
    recs = []
    for priority in range(max_steps):
        action = policy_fn(env)
        if action == env.n_tests:
            break
        if env._run_mask[action] == 1.0:
            continue
        tname = env.test_names[action]
        t = env.tests[tname]
        recs.append(TestRecommendation(
            test_name=tname,
            cost=t["cost"],
            time=t["time"],
            group=t["group"],
            priority=priority + 1,
        ))
        env.step(action)
        if env._done:
            break
    return recs
