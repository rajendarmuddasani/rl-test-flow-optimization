# RL Test Flow Optimization

Reinforcement learning system for semiconductor post-silicon test flow optimization. Trains PPO, MaskablePPO, DQN, and A2C agents to select optimal test sequences that maximize defect detection while minimizing cost and time.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
│                                                              │
│  generate_test_data.py ──→ data/chip_test_data.parquet       │
│        (1M chips × 1000 tests)                               │
│                                                              │
│  ┌─────────────────────────────┐  ┌────────────────────────┐ │
│  │  TestFlowEnv (Gymnasium)    │  │  Agent Training        │ │
│  │  • 10–1000 test actions     │  │  • MaskablePPO (best)  │ │
│  │  • Action masking           │  │  • PPO / DQN / A2C     │ │
│  │  • Cost/time budgets        │  │  • Optuna HPO (50 tr)  │ │
│  │  • 14 defect categories     │  │  • MLflow tracking     │ │
│  └─────────────────────────────┘  └────────────────────────┘ │
│                                                              │
│  ┌─────────────────────────────┐  ┌────────────────────────┐ │
│  │  3-Stage Training Funnel    │  │  Curriculum Learning   │ │
│  │  Stage 1: 4 algos, 200K    │  │  10 → 50 → 200 tests  │ │
│  │  Stage 2: top-2, 500K, 3s  │  │  Progressive scaling   │ │
│  │  Stage 3: best, 1M steps   │  │                        │ │
│  └─────────────────────────────┘  └────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                    Inference Service                         │
│                                                              │
│  FastAPI (port 8000) ──→ POST /optimize                      │
│       │                  GET  /health                         │
│       │                  GET  /tests                          │
│       └── Trained model or greedy heuristic fallback         │
│                                                              │
│  Docker Compose: api + MLflow (port 5000)                    │
└──────────────────────────────────────────────────────────────┘
```

## Key Features

- **Scalable environment**: 10 to 1000 tests with action masking for invalid actions
- **4 RL algorithms**: MaskablePPO, PPO, DQN, A2C via Stable-Baselines3
- **3 heuristic baselines**: Random, greedy coverage, cost-efficient
- **3-stage training funnel**: Broad search → deep comparison → final optimization
- **Optuna HPO**: 50-trial hyperparameter search with pruning
- **Curriculum learning**: Progressive test count scaling (10 → 50 → 200 → 1000)
- **14 defect categories**: Electrical, timing, thermal, functional, analog
- **1M-chip data generator**: Parquet output with chunked processing

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Generate data (10K chips for local testing)
python generate_test_data.py --chips 10000 --tests 100

# Train agent
python -m src.cli train --algo maskable_ppo --timesteps 50000 --n-tests 10

# Evaluate
python -m src.cli evaluate --n-tests 10

# Demo episodes
python -m src.cli demo --n-tests 10

# Start API server
python -m src.cli serve
```

## GPU Training (Kaggle/Colab)

Upload `notebooks/train_rl_kaggle.ipynb` to Kaggle with T4 GPU enabled:

1. Generates 100K chips × 200 tests
2. Trains all 4 algorithms (200K steps each)
3. Deep training on top-2 (500K steps, 3 seeds)
4. Optuna HPO (30 trials)
5. Final training with best params (1M steps)
6. Curriculum training (10 → 50 → 200 tests)
7. Exports 12+ publication-quality plots

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/optimize` | Get optimal test sequence for a chip |
| GET | `/health` | Service health check |
| GET | `/tests?n_tests=10` | List available tests |
| POST | `/load-model` | Load a trained model for inference |

### Example Request

```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"chip_id": "CHIP_001", "n_tests": 10, "cost_budget": 30}'
```

## Project Structure

```
rl-test-flow-optimization/
├── src/
│   ├── environment.py          # Gymnasium env with action masking
│   ├── agent.py                # 4 algos, baselines, Optuna HPO
│   ├── api.py                  # FastAPI inference service
│   └── cli.py                  # Click CLI (generate/train/evaluate/demo/serve)
├── generate_test_data.py       # 1M-chip data generator (Parquet)
├── notebooks/
│   └── train_rl_kaggle.ipynb   # Full GPU training pipeline
├── tests/
│   ├── test_environment.py     # Env tests (action masking, scaling)
│   └── test_data.py            # Data generator tests
├── docker-compose.yml          # API + MLflow services
├── Dockerfile                  # API container
├── pyproject.toml              # Dependencies (SB3, sb3-contrib, Optuna, etc.)
└── requirements.txt
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| RL Framework | Stable-Baselines3, sb3-contrib |
| Environment | Gymnasium (custom) |
| HPO | Optuna |
| Experiment Tracking | MLflow |
| Data Format | Parquet (PyArrow) |
| API | FastAPI + Uvicorn |
| Containerization | Docker, Docker Compose |
| GPU Training | Kaggle T4 / Colab A100 |

## Defect Taxonomy

| Group | Defect Types |
|-------|-------------|
| Electrical | voltage_droop, current_leak, esd_damage |
| Timing | setup_violation, hold_violation, clock_jitter, frequency_fail |
| Thermal | power_thermal, electromigration |
| Functional | logic_fail, memory_fail, io_fail, scan_fail |
| Analog | analog_drift |

## License

MIT
