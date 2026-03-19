"""Dataset balancing via home/away symmetry swapping.

Analogous to UFC fighter A/B swapping -- for each game, we can view it
from either team's perspective. This doubles the dataset and ensures the
model doesn't learn a bias toward home or away.
"""

import pandas as pd


def create_symmetric_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Create a balanced dataset by swapping home/away perspectives.

    For each game row with features like diff_X = home_X - away_X,
    create a mirror row where diff_X = away_X - home_X and the
    target is flipped.

    Args:
        df: DataFrame with diff_ and ratio_ columns plus 'home_win' target.

    Returns:
        DataFrame with original + mirrored rows, shuffled.
    """
    mirror = df.copy()

    # Flip diff features
    diff_cols = [c for c in df.columns if c.startswith("diff_")]
    for col in diff_cols:
        mirror[col] = -mirror[col]

    # Flip ratio features (invert)
    ratio_cols = [c for c in df.columns if c.startswith("ratio_")]
    for col in ratio_cols:
        mirror[col] = mirror[col].apply(lambda x: 1 / x if x != 0 else 0)

    # Flip target
    if "home_win" in mirror.columns:
        mirror["home_win"] = 1 - mirror["home_win"]

    # Flip the is_home indicator if present
    if "is_home" in mirror.columns:
        mirror["is_home"] = 1 - mirror["is_home"]

    combined = pd.concat([df, mirror], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return combined
