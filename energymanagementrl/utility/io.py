import os

import pandas as pd


def save_with_suffix(
    df: pd.DataFrame,
    directory: str,
    basename: str,
    suffix: str | None = None,
    index_label: str = "timestamp",
    **to_csv_kwargs,
):
    os.makedirs(directory, exist_ok=True)
    df.to_csv(
        os.path.join(directory, f"{basename}.csv"),
        index_label=index_label,
        **to_csv_kwargs,
    )
    if suffix:
        df.to_csv(
            os.path.join(directory, f"{basename}.{suffix}.csv"),
            index_label=index_label,
            **to_csv_kwargs,
        )
