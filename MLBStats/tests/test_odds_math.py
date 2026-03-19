"""Tests for odds math utilities."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.odds_math import (
    american_to_decimal, decimal_to_american, american_to_implied_prob,
    remove_vig, compute_ev, kelly_fraction,
)


def test_american_to_decimal():
    assert abs(american_to_decimal(-150) - 1.6667) < 0.01
    assert abs(american_to_decimal(130) - 2.30) < 0.01
    assert abs(american_to_decimal(100) - 2.0) < 0.01
    assert abs(american_to_decimal(-100) - 2.0) < 0.01


def test_american_to_implied_prob():
    assert abs(american_to_implied_prob(-150) - 0.6) < 0.01
    assert abs(american_to_implied_prob(130) - 0.4348) < 0.01
    assert abs(american_to_implied_prob(-110) - 0.5238) < 0.01


def test_remove_vig():
    h, a = remove_vig(0.5238, 0.5238)
    assert abs(h - 0.5) < 0.01
    assert abs(a - 0.5) < 0.01


def test_compute_ev():
    ev = compute_ev(0.55, 2.0)  # 55% chance at even money
    assert ev > 0

    ev = compute_ev(0.45, 2.0)
    assert ev < 0


def test_kelly_fraction():
    k = kelly_fraction(0.55, 2.0, fraction=1.0)  # full Kelly
    assert abs(k - 0.10) < 0.01

    k = kelly_fraction(0.45, 2.0, fraction=1.0)
    assert k == 0.0  # negative EV, no bet


if __name__ == "__main__":
    test_american_to_decimal()
    test_american_to_implied_prob()
    test_remove_vig()
    test_compute_ev()
    test_kelly_fraction()
    print("All tests passed!")
