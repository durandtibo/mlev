r"""Contain utility functions for the missing value policy."""

from __future__ import annotations

__all__ = ["MISSING_POLICIES", "check_missing_policy"]


MISSING_POLICIES = ["omit", "propagate", "raise"]


def check_missing_policy(missing_policy: str) -> None:
    r"""Check the missing value policy.

    Args:
        missing_policy: The missing value policy.

    Raises:
        ValueError: if ``missing_policy`` is not ``'omit'``,
            ``'propagate'``, or ``'raise'``.

    Example:
        ```pycon
        >>> from mlev.utils.missing import check_missing_policy
        >>> check_missing_policy(missing_policy="omit")

        ```
    """
    if missing_policy not in {"omit", "propagate", "raise"}:
        msg = (
            f"Incorrect 'missing_policy': {missing_policy}. The valid values are: "
            f"'omit', 'propagate', 'raise'"
        )
        raise ValueError(msg)
