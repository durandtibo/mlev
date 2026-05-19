r"""Utilities to preprocess ``polars.DataFrame`` with missing values."""

from __future__ import annotations

__all__ = ["preprocess"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl


def preprocess(frame: pl.DataFrame, drop_missing: bool = False) -> pl.DataFrame:
    r"""Preprocess a DataFrame by optionally removing rows with null
    values.

    Args:
        frame: The DataFrame to preprocess.
        drop_missing: If ``True``, drop rows where any column value is
            null; otherwise return the input frame unchanged.

    Returns:
        A preprocessed DataFrame.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from mlev.utils.frame import preprocess
        >>> frame = pl.DataFrame({"y_true": [1, 0, 0, 1, 1, None], "y_pred": [0, 1, 0, 1, None, 1]})
        >>> preprocess(frame)
        shape: (6, 2)
        ┌────────┬────────┐
        │ y_true ┆ y_pred │
        │ ---    ┆ ---    │
        │ i64    ┆ i64    │
        ╞════════╪════════╡
        │ 1      ┆ 0      │
        │ 0      ┆ 1      │
        │ 0      ┆ 0      │
        │ 1      ┆ 1      │
        │ 1      ┆ null   │
        │ null   ┆ 1      │
        └────────┴────────┘
        >>> preprocess(frame, drop_missing=True)
        shape: (4, 2)
        ┌────────┬────────┐
        │ y_true ┆ y_pred │
        │ ---    ┆ ---    │
        │ i64    ┆ i64    │
        ╞════════╪════════╡
        │ 1      ┆ 0      │
        │ 0      ┆ 1      │
        │ 0      ┆ 0      │
        │ 1      ┆ 1      │
        └────────┴────────┘

        ```
    """
    if not drop_missing:
        return frame
    return frame.drop_nulls()
