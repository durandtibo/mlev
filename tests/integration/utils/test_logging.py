from __future__ import annotations

from mlev.testing.fixtures import rich_available, rich_not_available
from mlev.utils.logging import log_markdown

##################################
#     Tests for log_markdown     #
##################################


@rich_available
def test_log_markdown_with_rich() -> None:
    log_markdown("# Hello")


@rich_not_available
def test_log_markdown_without_rich() -> None:
    log_markdown("# Hello")
