from collections import Counter

import pandas as pd

from ..utility import get_logger

logger = get_logger(__name__)

DAY = 288


def generate_gap_report(df: pd.DataFrame):
    n_nan_total = df.isna().sum().sum()
    if n_nan_total == 0:
        logger.info("No missing values found")
        return

    logger.info(f"Total NaN cells: {n_nan_total}")
    for col in df.columns:
        n = df[col].isna().sum()
        if n:
            pct = n / len(df) * 100
            logger.info(f"  {col}: {n} NaN ({pct:.2f}%)")

    any_nan = df.isna().any(axis=1)
    gaps = []
    start = None
    for i, is_nan in enumerate(any_nan):
        if is_nan and start is None:
            start = i
        elif not is_nan and start is not None:
            gaps.append((df.index[start], df.index[i - 1], i - start))
            start = None
    if start is not None:
        gaps.append((df.index[start], df.index[-1], len(df) - start))

    logger.info(f"Total gap blocks: {len(gaps)}")
    if not gaps:
        return

    dist = Counter(g[2] for g in gaps)
    logger.info("Gap size distribution:")
    for size, count in sorted(dist.items(), key=lambda x: -x[1])[:10]:
        days = size / DAY
        logger.info(f"  {size:>5} rows ({days:.2f} days): {count} occurrences")

    logger.info("10 largest gaps:")
    for frm, to, length in sorted(gaps, key=lambda g: -g[2])[:10]:
        days = length / DAY
        logger.info(
            f"  {frm:%Y-%m-%d %H:%M}  \u2192  {to:%Y-%m-%d %H:%M}  ({length} rows, {days:.2f} days)"
        )
