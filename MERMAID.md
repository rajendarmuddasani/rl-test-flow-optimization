# rl-test-flow-optimization — End-to-End Flow

Every step of the pipeline, top to bottom. Dark canvas, color-coded by stage:
data (teal), environment (purple), algorithms (orange), training (red), HPO (yellow),
tracking (blue), serving (green).

**Rendered assets** (alongside this file):
- `MERMAID_flow.png` — full-resolution static render (1800 × 13223 px)
- `MERMAID_flow.gif` — slow top-to-bottom scrolling animation (~11 s, 720 px wide)

![Scrolling flow](./MERMAID_flow.gif)

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "background": "#0b0b0f",
    "primaryColor": "#1f2937",
    "primaryTextColor": "#ffffff",
    "primaryBorderColor": "#ffffff",
    "lineColor": "#ffffff",
    "textColor": "#ffffff",
    "fontSize": "16px",
    "fontFamily": "Inter, Arial, sans-serif",
    "clusterBkg": "#111827",
    "clusterBorder": "#e5e7eb"
  }
}}%%
flowchart TD

%% ═══════════════════════ 0. START ═══════════════════════
S([▶ START — RL Test Flow Optimization])
S --> ENV0

%% ═══════════════════════ 1. ENVIRONMENT SETUP ═══════════════════════
subgraph ENVSETUP [" 1. Environment Setup "]
  direction TB
  ENV0[Clone repo<br/>git clone rl-test-flow-optimization]
  ENV1[Copy .env.example → .env]
  ENV2[Edit ALGORITHM, TIMESTEPS, SEED,<br/>MLFLOW_TRACKING_URI, N_ENVS, N_STEPS]
  ENV3[pip install -e .[dev]]
  ENV4[Verify: gymnasium, stable-baselines3,<br/>sb3-contrib, optuna, mlflow, fastapi]
  ENV0 --> ENV1 --> ENV2 --> ENV3 --> ENV4
end
ENV4 --> DG0

%% ═══════════════════════ 2. DATA GENERATION ═══════════════════════
subgraph DATAGEN [" 2. Data Generation — generate_test_data.py "]
  direction TB
  DG0[Seed RNG with SEED=42]
  DG1[Pick N_CHIPS = 1,000,000<br/>Pick N_TESTS = 1,000]
  DG2[Draw chip features<br/>voltage margin, process corner,<br/>temperature, wafer position]
  DG3[Inject defects at DEFECT_RATE=0.70<br/>across 14 categories in 5 groups]
  DG4{{14 defect categories:<br/>voltage_droop, current_leak, esd,<br/>setup/hold violation, clock_jitter,<br/>freq_fail, power_thermal, EM,<br/>logic/memory/io/scan_fail, analog_drift}}
  DG5[Build test catalog<br/>TEST_0000..TEST_0999<br/>cost, time, group, coverage]
  DG6[Simulate pass/fail per test per chip<br/>using coverage probability]
  DG7[Chunked write: 10k rows/chunk<br/>→ data/chip_test_data.parquet]
  DG8[Stratified split<br/>800k train / 100k val / 100k test]
  DG0 --> DG1 --> DG2 --> DG3 --> DG4 --> DG5 --> DG6 --> DG7 --> DG8
end
DG8 --> GE0

%% ═══════════════════════ 3. GYMNASIUM ENVIRONMENT ═══════════════════════
subgraph GYMENV [" 3. Gymnasium Environment — TestFlowEnv "]
  direction TB
  GE0[Define observation space<br/>Box per-test: 5 features × N_TESTS<br/>Global: 3 features<br/>budget_frac, time_frac, tests_run_frac]
  GE1[Define action space<br/>Discrete: N_TESTS + 1 STOP action]
  GE2[reset: pick chip, set budget=100,<br/>time_budget=60, tests_run mask=∅]
  GE3[step: receive action a_t]
  GE4{Is test a_t<br/>already run<br/>OR over budget<br/>OR STOP?}
  GE5[Mask violation — large neg reward<br/>or terminate episode]
  GE6[Run test a_t, observe pass/fail]
  GE7[Reward =<br/>+1.0 × coverage_gained<br/>−0.1 × norm_cost<br/>−0.05 × norm_time]
  GE8[Update state<br/>mark test run, decrement budgets,<br/>update running coverage]
  GE9[action_masks method<br/>exposes valid action set to MaskablePPO]
  GE0 --> GE1 --> GE2 --> GE3 --> GE4
  GE4 -- "invalid" --> GE5
  GE4 -- "STOP" --> GE5
  GE4 -- "valid" --> GE6 --> GE7 --> GE8 --> GE3
  GE2 --> GE9
end
GE8 --> BL0

%% ═══════════════════════ 4. BASELINE POLICIES ═══════════════════════
subgraph BASELINES [" 4. Baseline Policies (no learning) "]
  direction TB
  BL0[Random policy]
  BL1[Greedy coverage policy<br/>argmax coverage ignoring cost]
  BL2[Cost-efficient heuristic<br/>argmax coverage / cost]
  BL3[Evaluate each: 10k episodes<br/>log coverage, tests_run, cost, time]
  BL0 --> BL3
  BL1 --> BL3
  BL2 --> BL3
end
BL3 --> ALG0

%% ═══════════════════════ 5. 4 RL ALGORITHMS ═══════════════════════
subgraph ALGS [" 5. Four RL Algorithms — ALGO_REGISTRY "]
  direction TB
  ALG0[Algorithm selector — --algo flag]
  A1[PPO<br/>on-policy, clipped surrogate]
  A2[MaskablePPO ⭐<br/>invalid-action mask enforced]
  A3[DQN<br/>off-policy, replay buffer]
  A4[A2C<br/>synchronous actor-critic]
  ALG0 --> A1
  ALG0 --> A2
  ALG0 --> A3
  ALG0 --> A4
end
A1 --> TR0
A2 --> TR0
A3 --> TR0
A4 --> TR0

%% ═══════════════════════ 6. 3-STAGE TRAINING FUNNEL ═══════════════════════
subgraph FUNNEL [" 6. 3-Stage Training Funnel "]
  direction TB
  TR0[train-funnel CLI entry]
  TR1[Stage 1 — Broad Search<br/>4 algos × 200k steps × 3 seeds<br/>12 parallel runs via N_ENVS=8]
  TR2[Rank by mean eval reward<br/>pick top-2]
  TR3[Stage 2 — Deep Comparison<br/>top-2 × 500k steps × 5 seeds]
  TR4[Rank by area-under-learning-curve<br/>pick winner]
  TR5[Stage 3 — Final Optimization<br/>winner × 1M steps × best seed]
  TR0 --> TR1 --> TR2 --> TR3 --> TR4 --> TR5
end
TR1 --> CL0
TR5 --> HPO0

%% ═══════════════════════ 7. CURRICULUM LEARNING ═══════════════════════
subgraph CURR [" 7. Curriculum Learning (wraps all stages) "]
  direction TB
  CL0[Start N_TESTS=10]
  CL1{Eval coverage<br/>> 70% for<br/>10 consecutive evals?}
  CL2[Grow N_TESTS: 10 → 50 → 200 → 1000]
  CL3[Transfer weights, reset optimizer state]
  CL4[Repeat Stage 1-3 at new scale]
  CL0 --> CL1
  CL1 -- "yes" --> CL2 --> CL3 --> CL4 --> CL1
  CL1 -- "no" --> CL0
end
CL4 --> HPO0

%% ═══════════════════════ 8. OPTUNA HPO ═══════════════════════
subgraph HPO [" 8. Optuna Hyperparameter Optimization "]
  direction TB
  HPO0[Create study in SQLite<br/>storage=sqlite:///optuna.db]
  HPO1[50 trials with MedianPruner<br/>prune after warmup]
  HPO2[Search space:<br/>lr ∈ [1e-5, 1e-3]<br/>ent_coef ∈ [0, 0.02]<br/>clip_range ∈ [0.1, 0.3]<br/>n_steps ∈ 1024/2048/4096<br/>gae_lambda ∈ [0.8, 1.0]]
  HPO3[Objective: mean reward over last 10 evals]
  HPO4[Persist best params<br/>Re-train Stage 3 with winner]
  HPO0 --> HPO1 --> HPO2 --> HPO3 --> HPO4
end
HPO4 --> MLF0

%% ═══════════════════════ 9. MLFLOW TRACKING ═══════════════════════
subgraph MLFLOW [" 9. MLflow Experiment Tracking "]
  direction TB
  MLF0[Start MLflow run per stage/trial]
  MLF1[Log params:<br/>algo, timesteps, seed,<br/>curriculum scale, HPO params]
  MLF2[Log metrics every 10k steps:<br/>ep_rew_mean, coverage, cost,<br/>value_loss, policy_loss, kl_div]
  MLF3[Log artifacts:<br/>best_model.zip, eval videos,<br/>learning curves, action histograms]
  MLF4[Register best model<br/>mlflow.pyfunc.log_model]
  MLF0 --> MLF1 --> MLF2 --> MLF3 --> MLF4
end
MLF4 --> EVAL0

%% ═══════════════════════ 10. EVALUATION ═══════════════════════
subgraph EVAL [" 10. Evaluation on 100k Held-out Chips "]
  direction TB
  EVAL0[Load models/best_model.zip]
  EVAL1[Run 100k episodes deterministically]
  EVAL2["Compute: coverage at budget, tests_run,<br/>time_ms, cost, defect recall per category"]
  EVAL3[Compare vs 3 baselines<br/>t-test significance]
  EVAL4{"Coverage at budget<br/>≥ 75%?"}
  EVAL5[Promote to production<br/>tag v1.0.0]
  EVAL6[Log regression<br/>trigger retrain]
  EVAL0 --> EVAL1 --> EVAL2 --> EVAL3 --> EVAL4
  EVAL4 -- "yes" --> EVAL5
  EVAL4 -- "no" --> EVAL6
end
EVAL5 --> API0

%% ═══════════════════════ 11. FASTAPI SERVING ═══════════════════════
subgraph SERVE [" 11. FastAPI Inference Service "]
  direction TB
  API0[src.api:app — uvicorn bind 0.0.0.0:8000]
  API1[Load MODEL_CHECKPOINT on boot<br/>preload weights → GPU or CPU]
  API2[Middleware: auth via API_KEY header,<br/>rate limit 60/min, request logging]
  API3[POST /optimize<br/>body: chip_features, budget, time_budget]
  API4[Convert to observation tensor<br/>enforce action mask]
  API5[policy.predict deterministic=True]
  API6[Iterate until STOP or budget exhausted]
  API7[Return JSON:<br/>test_sequence, predicted_coverage,<br/>cost_used, time_used]
  API8[GET /health — liveness]
  API9[GET /metrics — Prometheus]
  API0 --> API1 --> API2 --> API3 --> API4 --> API5 --> API6 --> API7
  API2 --> API8
  API2 --> API9
end
API7 --> DEP0

%% ═══════════════════════ 12. DEPLOYMENT ═══════════════════════
subgraph DEPLOY [" 12. Deployment & Ops "]
  direction TB
  DEP0[docker build -t rl-test-flow:1.0.0]
  DEP1[docker compose up: api + mlflow]
  DEP2[Push image: ghcr.io/org/rl-test-flow:1.0.0]
  DEP3[kubectl apply deployment + HPA<br/>min=2 max=10 target CPU 70%]
  DEP4[PVC mounts /models<br/>best_model.zip read-only]
  DEP5[Liveness /health — readiness /health]
  DEP6[Prometheus scrapes /metrics<br/>Grafana dashboards]
  DEP7[SLOs:<br/>p99 latency < 150ms<br/>coverage ≥ 75% rolling 7d<br/>error < 0.5%]
  DEP0 --> DEP1 --> DEP2 --> DEP3 --> DEP4 --> DEP5 --> DEP6 --> DEP7
end
DEP7 --> END0

%% ═══════════════════════ 13. MONITORING LOOP ═══════════════════════
subgraph MON [" 13. Continuous Monitoring & Retrain "]
  direction TB
  END0[Shadow traffic splits 5% to new model]
  END1{Coverage drop<br/>> 5% WoW?}
  END2[Trigger retrain job<br/>fresh data gen → funnel → HPO]
  END3[Canary 1% → 10% → 100%]
  END4[Hold at current prod]
  END0 --> END1
  END1 -- "yes" --> END2 --> END3
  END1 -- "no" --> END4
end
END3 --> FIN
END4 --> FIN
FIN([■ END — policy serving live])

%% ═══════════════════════ STYLING ═══════════════════════
classDef start fill:#10b981,stroke:#ffffff,stroke-width:2px,color:#000000,font-weight:bold
classDef dataNode fill:#14b8a6,stroke:#ffffff,stroke-width:1.5px,color:#000000
classDef envNode fill:#a78bfa,stroke:#ffffff,stroke-width:1.5px,color:#000000
classDef algoNode fill:#fb923c,stroke:#ffffff,stroke-width:1.5px,color:#000000
classDef trainNode fill:#ef4444,stroke:#ffffff,stroke-width:1.5px,color:#ffffff
classDef hpoNode fill:#facc15,stroke:#ffffff,stroke-width:1.5px,color:#000000
classDef trackNode fill:#3b82f6,stroke:#ffffff,stroke-width:1.5px,color:#ffffff
classDef evalNode fill:#ec4899,stroke:#ffffff,stroke-width:1.5px,color:#ffffff
classDef serveNode fill:#22c55e,stroke:#ffffff,stroke-width:1.5px,color:#000000
classDef deployNode fill:#0ea5e9,stroke:#ffffff,stroke-width:1.5px,color:#ffffff
classDef decision fill:#f59e0b,stroke:#ffffff,stroke-width:2px,color:#000000

class S,FIN start
class DG0,DG1,DG2,DG3,DG4,DG5,DG6,DG7,DG8 dataNode
class ENV0,ENV1,ENV2,ENV3,ENV4,GE0,GE1,GE2,GE3,GE5,GE6,GE7,GE8,GE9 envNode
class BL0,BL1,BL2,BL3,A1,A2,A3,A4,ALG0 algoNode
class TR0,TR1,TR2,TR3,TR4,TR5,CL0,CL2,CL3,CL4 trainNode
class HPO0,HPO1,HPO2,HPO3,HPO4 hpoNode
class MLF0,MLF1,MLF2,MLF3,MLF4 trackNode
class EVAL0,EVAL1,EVAL2,EVAL3,EVAL5,EVAL6 evalNode
class API0,API1,API2,API3,API4,API5,API6,API7,API8,API9 serveNode
class DEP0,DEP1,DEP2,DEP3,DEP4,DEP5,DEP6,DEP7,END0,END2,END3,END4 deployNode
class GE4,EVAL4,CL1,END1 decision
```
