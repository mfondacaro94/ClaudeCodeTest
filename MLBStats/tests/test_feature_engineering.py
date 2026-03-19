"""Tests for feature engineering utilities."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from utils.data_cleaning import normalize_team_name, parse_innings_pitched, safe_float


def test_normalize_team_name():
    assert normalize_team_name("LAD") == "LAD"
    assert normalize_team_name("ANA") == "LAA"
    assert normalize_team_name("FLA") == "MIA"
    assert normalize_team_name("WSH") == "WSN"
    assert normalize_team_name("TBD") == "TBR"


def test_parse_innings_pitched():
    assert abs(parse_innings_pitched("6.1") - 6.333) < 0.01
    assert abs(parse_innings_pitched("6.2") - 6.667) < 0.01
    assert parse_innings_pitched("6.0") == 6.0
    assert parse_innings_pitched("") == 0.0


def test_safe_float():
    assert safe_float("3.14") == 3.14
    assert safe_float("") is None
    assert safe_float(None) is None
    assert safe_float("abc") is None
    assert safe_float("abc", 0.0) == 0.0


if __name__ == "__main__":
    test_normalize_team_name()
    test_parse_innings_pitched()
    test_safe_float()
    print("All tests passed!")
