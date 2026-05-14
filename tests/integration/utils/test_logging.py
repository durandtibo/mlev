from __future__ import annotations

from mlev.testing.fixtures import rich_available, rich_not_available
from mlev.utils.logging import log_dict_pretty

#####################################
#     Tests for log_dict_pretty     #
#####################################


@rich_available
def test_log_dict_pretty_with_rich() -> None:
    log_dict_pretty({"hello": "world"})


@rich_not_available
def test_log_dict_pretty_without_rich() -> None:
    log_dict_pretty({"hello": "world"})
