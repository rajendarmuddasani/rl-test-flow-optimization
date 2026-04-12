"""CLI for semiconductor test flow RL optimization."""

import json
import logging

import click
import numpy as np

logging.basicConfig(level=logging.INFO)


@click.group()
def cli():
    """RL-based semiconductor test flow optimizer."""
    pass


@click.command()
@click.option("--chips", default=1_000_000, help="Number of chips to generate")
@click.option("--tests", default=1000, help="Number of test types")
@click.option("--defect-rate", default=0.70, help="Fraction of defective chips")
@click.option("--output-dir", default="data", help="Output directory")
@click.option("--fmt", default="parquet", type=click.Choice(["parquet", "csv"]))
def generate(chips: int, tests: int, defect_rate: float, output_dir: str, fmt: str):
    """Generate synthetic test data."""
    from generate_test_data import generate_dataset

    summary = generate_dataset(
        output_dir=output_dir,
        n_chips=chips,
        n_tests=tests,
        defect_rate=defect_rate,
        fmt=fmt,
    )
    click.echo(json.dumps(summary, indent=2))


@click.command()
@click.option("--algo", default="maskable_ppo",
              type=click.Choice(["maskable_ppo", "ppo", "dqn", "a2c"]))
@click.option("--timesteps", default=200_000, help="Training timesteps")
@click.option("--n-tests", default=10, help="Number of tests in environment")
@click.option("--seed", default=42, help="Random seed")
@click.option("--output-dir", default="outputs/models", help="Model output dir")
def train(algo: str, timesteps: int, n_tests: int, seed: int, output_dir: str):
    """Train an RL agent on the test flow environment."""
    from src.environment import TestFlowEnv
    from src.agent import ALGO_REGISTRY

    env = TestFlowEnv(n_tests=n_tests)
    train_fn = ALGO_REGISTRY[algo]
    model = train_fn(env, total_timesteps=timesteps, output_dir=output_dir, seed=seed)
    click.echo(f"Training complete. Model saved to {output_dir}/")


@click.command()
@click.option("--model-path", default="outputs/models/maskable_ppo.zip", help="Model path")
@click.option("--n-tests", default=10, help="Number of tests in environment")
@click.option("--episodes", default=200, help="Evaluation episodes")
def evaluate(model_path: str, n_tests: int, episodes: int):
    """Evaluate trained agent vs baselines."""
    from src.environment import TestFlowEnv
    from src.agent import BASELINES, evaluate_policy, evaluate_trained_model

    env = TestFlowEnv(n_tests=n_tests)

    # Baselines
    for name, policy_fn in BASELINES.items():
        click.echo(f"\n=== Baseline: {name} ===")
        metrics = evaluate_policy(env, policy_fn, episodes)
        click.echo(json.dumps(metrics, indent=2))

    # Trained agent
    try:
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(model_path, env=env)
        click.echo(f"\n=== Trained Agent ({model_path}) ===")
        metrics = evaluate_trained_model(env, model, episodes)
        click.echo(json.dumps(metrics, indent=2))
    except FileNotFoundError:
        click.echo(f"\nModel not found at {model_path}. Train first with: python -m src.cli train")


@click.command()
@click.option("--episodes", default=5, help="Number of episodes to run")
@click.option("--n-tests", default=10, help="Number of tests in environment")
def demo(episodes: int, n_tests: int):
    """Run demo episodes with greedy coverage policy."""
    from src.environment import TestFlowEnv
    from src.agent import greedy_coverage_policy

    env = TestFlowEnv(n_tests=n_tests)
    for ep in range(episodes):
        obs, _ = env.reset()
        click.echo(f"\n{'='*60}")
        click.echo(f"Episode {ep + 1} | Defect: {env._defect or 'NONE (good chip)'}")
        click.echo(f"{'='*60}")
        done, step = False, 0
        while not done:
            action = greedy_coverage_policy(env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            action_name = env.test_names[action] if action < env.n_tests else "STOP"
            click.echo(f"  Step {step}: {action_name:20s} | reward={reward:+.2f} | {info}")
            step += 1


@click.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000, type=int)
def serve(host: str, port: int):
    """Start FastAPI inference server."""
    import uvicorn

    uvicorn.run("src.api:app", host=host, port=port, reload=False)


cli.add_command(generate)
cli.add_command(train)
cli.add_command(evaluate)
cli.add_command(demo)
cli.add_command(serve)

if __name__ == "__main__":
    cli()
