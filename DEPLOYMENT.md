# Deployment Guide — RL Test Flow Optimization

## Architecture

```
┌────────────────┐      ┌────────────────┐      ┌─────────────────┐
│  Test Runner   │─────▶│  FastAPI       │─────▶│  MaskablePPO    │
│  (caller)      │◀─────│  /optimize     │◀─────│  (SB3 policy)   │
└────────────────┘      └────────────────┘      └─────────────────┘
                               │
                               │ logs metrics / runs
                               ▼
                        ┌────────────────┐
                        │  MLflow        │
                        │  (port 5000)   │
                        └────────────────┘
```

## Local Development

```bash
git clone <repo>
cd rl-test-flow-optimization
cp .env.example .env                        # edit ALGORITHM, TIMESTEPS, MLFLOW_TRACKING_URI
pip install -e ".[dev]"
python generate_test_data.py --chips 10000  # small local dataset
python -m src.cli train --algorithm maskable_ppo --timesteps 50000 --n-tests 100
uvicorn src.api:app --reload --port 8000
```

Smoke test:

```bash
curl -X POST http://localhost:8000/optimize \
  -H 'Content-Type: application/json' \
  -d '{"chip_features": [0.1, 0.2, 0.3], "budget": 100, "time_budget": 60}'
```

## Docker Compose (recommended for staging)

```bash
docker compose up -d
```

Brings up:
- `api` — FastAPI inference service on `:8000`
- `mlflow` — tracking server on `:5000`

Mounts `./models` and `./data` as volumes so trained checkpoints persist.

## Kubernetes

Minimal deployment manifest (`k8s/deployment.yaml`):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rl-test-flow-api
spec:
  replicas: 3
  selector:
    matchLabels: { app: rl-test-flow-api }
  template:
    metadata:
      labels: { app: rl-test-flow-api }
    spec:
      containers:
      - name: api
        image: ghcr.io/<org>/rl-test-flow:1.0.0
        ports: [{ containerPort: 8000 }]
        env:
        - name: MODEL_CHECKPOINT
          value: /models/best_model.zip
        - name: MLFLOW_TRACKING_URI
          value: http://mlflow.mlops:5000
        resources:
          requests: { cpu: "500m", memory: "1Gi" }
          limits:   { cpu: "2",    memory: "4Gi" }
        livenessProbe:
          httpGet: { path: /health, port: 8000 }
          initialDelaySeconds: 30
        readinessProbe:
          httpGet: { path: /health, port: 8000 }
        volumeMounts:
        - { name: models, mountPath: /models, readOnly: true }
      volumes:
      - { name: models, persistentVolumeClaim: { claimName: rl-models-pvc } }
```

HPA on CPU (target 70%) with `minReplicas: 2`, `maxReplicas: 10`.

## Cloud Training (GPU)

### Colab / Kaggle

Use `notebooks/NB04_production_training_curriculum_funnel.ipynb`. Set `DRIVE_PATH` for checkpoint persistence and `MLFLOW_TRACKING_URI` for run logging.

### AWS SageMaker

```python
from sagemaker.pytorch import PyTorch
estimator = PyTorch(
    entry_point="src/cli.py",
    source_dir=".",
    role=role,
    instance_type="ml.p4d.24xlarge",   # A100 ×8
    framework_version="2.1",
    py_version="py310",
    hyperparameters={"mode": "train-funnel", "algorithm": "auto", "timesteps": 1700000},
)
estimator.fit({"train": "s3://<bucket>/rl-test-flow/data/"})
```

## Configuration

Set via `.env` (see `.env.example`):

| Var | Default | Purpose |
|---|---|---|
| `ALGORITHM` | `maskable_ppo` | RL algorithm to train |
| `TIMESTEPS` | `50000` | Total training timesteps |
| `N_TESTS` | `10` | Test catalog size for this run |
| `SEED` | `42` | Determinism seed |
| `LEARNING_RATE` | `3e-4` | Optimizer LR |
| `BATCH_SIZE` | `64` | Mini-batch size |
| `N_ENVS` | `8` | Parallel envs for training |
| `N_STEPS` | `2048` | Rollout length per env |
| `OPTUNA_N_TRIALS` | `30` | HPO trials for stage 3 |
| `MODEL_CHECKPOINT` | `models/best_model.zip` | Path served by API |
| `API_HOST`, `API_PORT` | `0.0.0.0`, `8000` | FastAPI bind |

## Monitoring

- **Health**: `GET /health` — liveness probe.
- **Metrics**: Prometheus metrics at `/metrics` (request rate, latency histogram, cache hit rate).
- **MLflow**: Every training run logs params, metrics, and the best model artifact. Dashboard at `MLFLOW_TRACKING_URI`.
- **SLOs**:
  - p99 `/optimize` latency < 150 ms
  - Coverage@budget >= 75% (rolling 7-day on shadow traffic)
  - Error rate < 0.5%

## Scaling Notes

- **CPU-bound inference.** Single policy pass is ~20 ms on `c7i.xlarge`. Scale horizontally with HPA.
- **Stateless service.** No per-request state; safe to autoscale aggressively.
- **Warm model cache.** First request after boot is ~300 ms (torch import + weight load). Use `--preload` with `gunicorn` in prod.

## Security

- `API_KEY` header required for `/optimize` in production (set via `.env`).
- Rate-limit at ingress (60 RPM per key is a reasonable default).
- Model checkpoints are signed; API refuses to load any `.zip` not in the allow-list.

## Runbook

| Symptom | Likely cause | Action |
|---|---|---|
| 503 with "model not loaded" | `MODEL_CHECKPOINT` missing | mount PVC, restart pod |
| Latency p99 > 500 ms | CPU throttling | raise limits, check HPA |
| Coverage drop > 5% week-over-week | Data drift | trigger shadow retrain + eval |
| MLflow 502 | Tracking server down | training continues (best-effort); fix server |

## Rollback

```bash
kubectl rollout undo deployment/rl-test-flow-api
# or pin a specific image
kubectl set image deployment/rl-test-flow-api api=ghcr.io/<org>/rl-test-flow:0.9.3
```

Previous models are retained in the `mlflow-artifacts` bucket for 90 days.
