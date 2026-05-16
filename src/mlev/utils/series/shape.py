r"""Utilities to validate series shapes and dimensions."""

from __future__ import annotations

__all__ = ["check_same_shape"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    import polars as pl


def check_same_shape(series: Iterable[pl.Series]) -> None:
    r"""Check if series have the same shape.

    Args:
        series: The series to check.

    Raises:
        RuntimeError: if the series have different shapes.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from mlev.utils.series import check_same_shape
    >>> check_same_shape([pl.Series("col", [1, 0, 0, 1]), pl.Series("col", [0, 1, 0, 1])])

    ```
    """
    shapes = {arr.shape for arr in series}
    if len(shapes) > 1:
        msg = f"series have different shapes: {shapes}"
        raise RuntimeError(msg)
