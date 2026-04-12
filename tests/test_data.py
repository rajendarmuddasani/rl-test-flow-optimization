"""Tests for RL test flow data generator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataGenerator:
    def test_build_test_catalog(self):
        from generate_test_data import _build_test_catalog
        catalog = _build_test_catalog(100, seed=42)
        assert len(catalog) == 100
        for name, spec in catalog.items():
            assert "cost" in spec
            assert "time" in spec
            assert "group" in spec
            assert "defect_coverage" in spec
            assert spec["cost"] > 0
            assert spec["time"] > 0
            assert 0 < spec["defect_coverage"] <= 1

    def test_defect_categories(self):
        from generate_test_data import DEFECT_CATEGORIES
        assert "voltage_droop" in DEFECT_CATEGORIES
        assert "no_defect" in DEFECT_CATEGORIES
        assert len(DEFECT_CATEGORIES) == 15

    def test_category_groups(self):
        from generate_test_data import _CATEGORY_GROUPS
        assert "electrical" in _CATEGORY_GROUPS
        assert "timing" in _CATEGORY_GROUPS
        assert len(_CATEGORY_GROUPS) == 5

    def test_assign_defect(self):
        import numpy as np
        from generate_test_data import _assign_defect
        rng = np.random.RandomState(42)
        defects = [_assign_defect(rng, 0.70) for _ in range(1000)]
        assert "no_defect" in defects
        no_defect_rate = sum(1 for d in defects if d == "no_defect") / len(defects)
        assert 0.2 < no_defect_rate < 0.4  # ~30% no_defect

    def test_generate_test_results(self):
        import numpy as np
        from generate_test_data import _generate_test_results, _build_test_catalog
        catalog = _build_test_catalog(10, seed=42)
        rng = np.random.RandomState(42)
        results = _generate_test_results("voltage_droop", catalog, rng)
        assert len(results) == 10
        assert all(v in (0, 1) for v in results.values())
