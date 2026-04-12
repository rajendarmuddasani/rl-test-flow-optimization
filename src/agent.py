"""
RL agent training and evaluation for semiconductor test flow optimization.

Supports four algorithms (PPO, MaskablePPO, DQN, A2C), heuristic baselines,
and Optuna hyperparameter optimization with MLflow tracking.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ── Heuristic baselines ───────────────────────────────────────────────────


def random_policy(env) -> int:
    """Uniform random selection from valid actions."""
    mask = env.action_masks()
    valid = np.where(mask)[0]
    return int(np.random.choice(valid))


def greedy_coverage_policy(env) -> int:
    """Pick the highest-coverage affordable test, then STOP."""
    mask = env.action_masks()
    best_idx, best_cov = env.n_tests, -1.0  # default STOP
    for i in range(env.n_tests):
        if mask[i]:
            cov = env.tests[env.test_names[i]]["defect_coverage"]
            if cov > best_cov:
                best_cov = cov
                best_idx = i
    return best_idx


def cost_efficient_policy(env) -> int:
    """Best coverage/cost ratio; stop after budget 40% spent."""
    if env._cost_spent > 0.4 * env.cost_budget:
        return env.n_tests  # STOP
    mask = env.action_masks()
    best_idx, best_ratio = env.n_tests, -1.0
    for i in range(env.n_tests):
        if mask[i]:
            t = env.tests[env.test_names[i]]
            ratio = t["defect_coverage"] / max(t["cost"], 0.01)
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = i
    return best_idx


BASELINES = {
    "random": random_policy,
    "greedy_coverage": greedy_coverage_policy,
    "cost_efficient": cost_efficient_policy,
}


def evaluate_policy(env, policy_fn, n_episodes: int = 200) -> dict:
    """Evaluate a heuristic policy over multiple episodes."""
    rewards, costs, accuracies, tests_counts = [], [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, ep_reward = False, 0.0
        while not done:
            action = policy_fn(env)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        if "cost" in info:
            costs.append(info["cost"])
            accuracies.append(1.0 if info.get("correct") else 0.0)
            tests_counts.append(info.get("tests_run", 0))
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_cost": float(np.mean(costs)) if costs else 0.0,
        "accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "mean_tests": float(np.mean(tests_counts)) if tests_counts else 0.0,
    }


# ── SB3 training functions ────────────────────────────────────────────────


def train_maskable_ppo(
    env,
    total_timesteps: int = 200_000,
    output_dir: str = "outputs/models",
    seed: int = 42,
    **kwargs,
):
    """Train MaskablePPO agent (sb3-contrib) with action masking."""
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import EvalCallback

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=kwargs.get("learning_rate", 3e-4),
        n_steps=kwargs.get("n_steps", 2048),
        batch_size=kwargs.get("batch_size", 64),
        n_epochs=kwargs.get("n_epochs", 10),
        gamma=kwargs.get("gamma", 0.99),
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,
        seed=seed,
    )

    eval_cb = EvalCallback(
        env,
        best_model_save_path=str(out / "best_maskable_ppo"),
        log_path=str(out / "logs"),
        eval_freq=5000,
        n_eval_episodes=20,
        deterministic=True,
    )
    model.learn(total_timesteps=total_timesteps, callback=eval_cb)
    model.save(str(out / "maskable_ppo"))
    logger.info("MaskablePPO saved to %s", out / "maskable_ppo.zip")
    return model


def train_ppo(
    env,
    total_timesteps: int = 200_000,
    output_dir: str = "outputs/models",
    seed: int = 42,
    **kwargs,
):
    """Train standard PPO agent."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=kwargs.get("learning_rate", 3e-4),
        n_steps=kwargs.get("n_steps", 2048),
        batch_size=kwargs.get("batch_size", 64),
        n_epochs=kwargs.get("n_epochs", 10),
        gamma=kwargs.get("gamma", 0.99),
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,
        seed=seed,
    )

    eval_cb = EvalCallback(
        env,
        best_model_save_path=str(out / "best_ppo"),
        log_path=str(out / "logs"),
        eval_freq=5000,
        n_eval_episodes=20,
        deterministic=True,
    )
    model.learn(total_timesteps=total_timesteps, callback=eval_cb)
    model.save(str(out / "ppo"))
    logger.info("PPO saved to %s", out / "ppo.zip")
    return model


def train_dqn(
    env,
    total_timesteps: int = 200_000,
    output_dir: str = "outputs/models",
    seed: int = 42,
    **kwargs,
):
    """Train DQN agent."""
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import EvalCallback

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=kwargs.get("learning_rate", 1e-3),
        buffer_size=kwargs.get("buffer_size", 100_000),
        batch_size=kwargs.get("batch_size", 64),
        gamma=kwargs.get("gamma", 0.99),
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=0,
        seed=seed,
    )

    eval_cb = EvalCallback(
        env,
        best_model_save_path=str(out / "best_dqn"),
        log_path=str(out / "logs"),
        eval_freq=5000,
        n_eval_episodes=20,
        deterministic=True,
    )
    model.learn(total_timesteps=total_timesteps, callback=eval_cb)
    model.save(str(out / "dqn"))
    logger.info("DQN saved to %s", out / "dqn.zip")
    return model


def train_a2c(
    env,
    total_timesteps: int = 200_000,
    output_dir: str = "outputs/models",
    seed: int = 42,
    **kwargs,
):
    """Train A2C agent."""
    from stable_baselines3 import A2C
    from stable_baselines3.common.callbacks import EvalCallback

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=kwargs.get("learning_rate", 7e-4),
        n_steps=kwargs.get("n_steps", 5),
        gamma=kwargs.get("gamma", 0.99),
        gae_lambda=0.95,
        verbose=0,
        seed=seed,
    )

    eval_cb = EvalCallback(
        env,
        best_model_save_path=str(out / "best_a2c"),
        log_path=str(out / "logs"),
        eval_freq=5000,
        n_eval_episodes=20,
        deterministic=True,
    )
    model.learn(total_timesteps=total_timesteps, callback=eval_cb)
    model.save(str(out / "a2c"))
    logger.info("A2C saved to %s", out / "a2c.zip")
    return model


ALGO_REGISTRY = {
    "maskable_ppo": train_maskable_ppo,
    "ppo": train_ppo,
    "dqn": train_dqn,
    "a2c": train_a2c,
}


# ── Evaluation (trained models) ───────────────────────────────────────────


def evaluate_trained_model(env, model, n_episodes: int = 200) -> dict:
    """Evaluate a trained SB3 model and return metrics."""
    rewards, costs, accuracies, tests_counts = [], [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, ep_reward = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        if "cost" in info:
            costs.append(info["cost"])
            accuracies.append(1.0 if info.get("correct") else 0.0)
            tests_counts.append(info.get("tests_run", 0))
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_cost": float(np.mean(costs)) if costs else 0.0,
        "accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "mean_tests": float(np.mean(tests_counts)) if tests_counts else 0.0,
    }


# ── Optuna HPO ─────────────────────────────────────────────────────────────


def optuna_objective(trial, env_cls, env_kwargs: dict, algo: str = "ppo", timesteps: int = 50_000):
    """Optuna objective function for hyperparameter search."""
    import optuna

    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    env = env_cls(**env_kwargs)
    train_fn = ALGO_REGISTRY[algo]
    model = train_fn(
        env,
        total_timesteps=timesteps,
        output_dir=f"outputs/optuna/{trial.number}",
        learning_rate=lr,
        gamma=gamma,
        batch_size=batch_size,
    )

    metrics = evaluate_trained_model(env, model, n_episodes=50)
    return metrics["mean_reward"]


def run_optuna_hpo(
    env_cls,
    env_kwargs: dict,
    algo: str = "ppo",
    n_trials: int = 50,
    timesteps: int = 50_000,
) -> dict:
    """Run Optuna hyperparameter optimization study."""
    import optuna

    study = optuna.create_study(direction="maximize", study_name=f"hpo_{algo}")
    study.optimize(
        lambda trial: optuna_objective(trial, env_cls, env_kwargs, algo, timesteps),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
    }
