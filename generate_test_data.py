"""
Synthetic Post-Silicon Test Data Generator

Generates realistic semiconductor test data at configurable scale.
Supports Parquet output for large datasets (1M+ chips).

Defect categories based on real post-silicon validation failure modes:
  - Electrical (voltage, current, leakage, ESD)
  - Timing (setup, hold, clock, frequency)
  - Thermal (power, temperature, stress)
  - Functional (logic, memory, IO, scan)
  - None (pass-through)

Usage:
    python generate_test_data.py                    # 1M chips × 1000 tests
    python generate_test_data.py --chips 10000      # 10K chips
    python generate_test_data.py --tests 100        # 100 tests
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Test taxonomy
# ---------------------------------------------------------------------------

DEFECT_CATEGORIES = [
    "voltage_droop",
    "current_leak",
    "esd_damage",
    "setup_violation",
    "hold_violation",
    "clock_jitter",
    "frequency_fail",
    "power_thermal",
    "electromigration",
    "logic_fail",
    "memory_fail",
    "io_fail",
    "scan_fail",
    "analog_drift",
    "no_defect",
]

_CATEGORY_GROUPS = {
    "electrical": ["voltage_droop", "current_leak", "esd_damage"],
    "timing": ["setup_violation", "hold_violation", "clock_jitter", "frequency_fail"],
    "thermal": ["power_thermal", "electromigration"],
    "functional": ["logic_fail", "memory_fail", "io_fail", "scan_fail"],
    "analog": ["analog_drift"],
}


def _build_test_catalog(n_tests: int, seed: int = 42) -> dict:
    """Generate a catalog of n_tests semiconductor tests.

    Each test has: name, cost, time, group affinity, defect_coverage.
    """
    rng = np.random.RandomState(seed)
    groups = list(_CATEGORY_GROUPS.keys())
    catalog = {}

    for i in range(n_tests):
        group = groups[i % len(groups)]
        # Cost: 1-20 units, time: 2-60 units
        cost = round(float(rng.uniform(1, 20)), 2)
        time_s = round(float(rng.uniform(2, 60)), 2)
        # Defect coverage: higher for specialized tests
        coverage = round(float(rng.uniform(0.05, 0.40)), 3)
        catalog[f"TEST_{i:04d}"] = {
            "cost": cost,
            "time": time_s,
            "group": group,
            "defect_coverage": coverage,
        }

    return catalog


def _assign_defect(rng: np.random.RandomState, defect_rate: float = 0.70) -> str:
    """Assign a defect category to a chip."""
    if rng.random() < (1 - defect_rate):
        return "no_defect"
    return rng.choice(DEFECT_CATEGORIES[:-1])


def _generate_test_results(
    defect: str,
    catalog: dict,
    rng: np.random.RandomState,
) -> dict[str, int]:
    """Simulate pass/fail for each test given a chip's defect type."""
    group_for_defect = None
    for g, members in _CATEGORY_GROUPS.items():
        if defect in members:
            group_for_defect = g
            break

    results = {}
    for tname, tinfo in catalog.items():
        if defect == "no_defect":
            # Good chip: 95% pass, 5% false fail
            results[tname] = int(rng.random() > 0.05)
        elif tinfo["group"] == group_for_defect:
            # Test in same group as defect: 80% detect (fail)
            results[tname] = int(rng.random() > 0.80)
        else:
            # Unrelated test: 70% pass, 30% cross-detect
            results[tname] = int(rng.random() > 0.30)

    return results


def generate_dataset(
    output_dir: str = "data",
    n_chips: int = 1_000_000,
    n_tests: int = 1000,
    defect_rate: float = 0.70,
    seed: int = 42,
    chunk_size: int = 50_000,
    fmt: str = "parquet",
) -> dict:
    """Generate semiconductor test dataset.

    Args:
        output_dir: Directory for output files.
        n_chips: Number of chips to simulate.
        n_tests: Number of test types.
        defect_rate: Fraction of chips with defects.
        seed: Random seed.
        chunk_size: Rows per processing chunk (controls memory).
        fmt: Output format — 'parquet' or 'csv'.

    Returns:
        Summary statistics dict.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    t0 = time.time()
    catalog = _build_test_catalog(n_tests, seed=seed)

    # Save test catalog
    with open(out / "test_config.json", "w") as f:
        json.dump(catalog, f, indent=2)

    # Generate in chunks to manage memory
    all_frames = []
    defect_counts: dict[str, int] = {}

    for start in range(0, n_chips, chunk_size):
        end = min(start + chunk_size, n_chips)
        rows = []
        for i in range(start, end):
            defect = _assign_defect(rng, defect_rate)
            defect_counts[defect] = defect_counts.get(defect, 0) + 1
            results = _generate_test_results(defect, catalog, rng)
            rows.append(
                {"chip_id": f"CHIP_{i:07d}", "defect_type": defect, **results}
            )
        chunk_df = pd.DataFrame(rows)
        all_frames.append(chunk_df)
        print(f"  Generated chips {start:,}–{end-1:,}")

    df = pd.concat(all_frames, ignore_index=True)

    # Write output
    if fmt == "parquet":
        df.to_parquet(out / "chip_test_data.parquet", index=False, engine="pyarrow")
    else:
        df.to_csv(out / "chip_test_data.csv", index=False)

    elapsed = time.time() - t0

    summary = {
        "n_chips": n_chips,
        "n_tests": n_tests,
        "defect_rate": defect_rate,
        "defect_counts": defect_counts,
        "format": fmt,
        "generation_time_sec": round(elapsed, 2),
        "file_size_mb": round(
            (out / f"chip_test_data.{fmt}").stat().st_size / 1e6, 2
        ),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDataset: {n_chips:,} chips × {n_tests:,} tests in {elapsed:.1f}s")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate semiconductor test data")
    parser.add_argument("--chips", type=int, default=1_000_000)
    parser.add_argument("--tests", type=int, default=1000)
    parser.add_argument("--defect-rate", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data")
    parser.add_argument("--format", type=str, default="parquet", choices=["parquet", "csv"])
    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output,
        n_chips=args.chips,
        n_tests=args.tests,
        defect_rate=args.defect_rate,
        seed=args.seed,
        fmt=args.format,
    )
