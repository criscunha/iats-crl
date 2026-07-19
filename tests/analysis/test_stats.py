"""Tests for the paired statistical protocol."""

from __future__ import annotations

import numpy as np

from iats.analysis.stats import (
    clopper_pearson_upper,
    holm_bonferroni,
    mann_whitney_u,
    ratio_bootstrap_ci,
)


def test_holm_rejects_small_and_holds_large():
    out = holm_bonferroni({"a": 0.001, "b": 0.5, "c": 0.9})
    assert out["a"] is True
    assert out["b"] is False and out["c"] is False


def test_mann_whitney_detects_shift_and_null():
    rng = np.random.default_rng(2)
    big = rng.normal(1.0, 0.3, size=80)
    small = rng.normal(0.0, 0.3, size=80)
    assert mann_whitney_u(big, small, "greater") < 0.001  # big > small
    assert mann_whitney_u(small, big, "greater") > 0.99  # wrong direction
    assert mann_whitney_u(small, big, "less") < 0.001  # small < big (lower tail)
    same_a = rng.normal(0.0, 0.3, size=80)
    same_b = rng.normal(0.0, 0.3, size=80)
    assert mann_whitney_u(same_a, same_b, "two-sided") > 0.05  # no shift


def test_ratio_bootstrap_ci_recovers_ratio():
    rng = np.random.default_rng(3)
    a = rng.normal(4.0, 0.2, size=200)
    b = rng.normal(2.0, 0.1, size=200)
    ratio, lo, hi = ratio_bootstrap_ci(a, b)
    assert abs(ratio - 2.0) < 0.05
    assert lo > 1.0  # CI excludes 1 -> a clearly larger than b


def test_clopper_pearson_upper_closed_form_and_monotone():
    # 0/300 errors: closed form 1 - 0.05**(1/300) ~= 0.00993, lands at delta=0.01.
    assert abs(clopper_pearson_upper(0, 300) - (1 - 0.05 ** (1 / 300))) < 1e-12
    assert clopper_pearson_upper(0, 300) <= 0.01
    # Monotone in the error count; each bound at least covers the point estimate.
    bounds = [clopper_pearson_upper(k, 300) for k in (0, 1, 3, 10)]
    assert all(x < y for x, y in zip(bounds, bounds[1:], strict=False))  # pairs
    assert clopper_pearson_upper(10, 300) > 10 / 300
