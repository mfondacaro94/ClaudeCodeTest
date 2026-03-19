"""Odds conversion utilities: American, decimal, implied probability."""


def american_to_decimal(american: float) -> float:
    if american > 0:
        return 1 + american / 100
    elif american < 0:
        return 1 + 100 / abs(american)
    return 1.0


def decimal_to_american(decimal_odds: float) -> float:
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    elif decimal_odds > 1.0:
        return round(-100 / (decimal_odds - 1))
    return 0


def american_to_implied_prob(american: float) -> float:
    if american > 0:
        return 100 / (american + 100)
    elif american < 0:
        return abs(american) / (abs(american) + 100)
    return 0.5


def decimal_to_implied_prob(decimal_odds: float) -> float:
    if decimal_odds > 0:
        return 1 / decimal_odds
    return 0.5


def remove_vig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Remove vigorish from two-outcome implied probabilities."""
    total = prob_a + prob_b
    if total == 0:
        return 0.5, 0.5
    return prob_a / total, prob_b / total


def compute_edge(model_prob: float, implied_prob: float) -> float:
    """Compute the edge: model probability minus implied probability."""
    return model_prob - implied_prob


def compute_ev(model_prob: float, decimal_odds: float) -> float:
    """Expected value of a $1 bet."""
    return model_prob * (decimal_odds - 1) - (1 - model_prob)


def kelly_fraction(model_prob: float, decimal_odds: float, fraction: float = 0.25) -> float:
    """Fractional Kelly criterion bet sizing.

    Args:
        model_prob: Model's estimated win probability.
        decimal_odds: Decimal odds offered.
        fraction: Kelly fraction (0.25 = quarter Kelly, conservative).

    Returns:
        Fraction of bankroll to bet (0 if negative EV).
    """
    b = decimal_odds - 1
    q = 1 - model_prob
    if b <= 0:
        return 0.0
    kelly = (model_prob * b - q) / b
    return max(0.0, kelly * fraction)
