from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from mlev.testing.fixtures import colorlog_available, rich_available
from mlev.utils.logging import configure_logging, log_dict_pretty

MODULE = "mlev.utils.logging"


@pytest.fixture(autouse=True)
def _reset_logging() -> None:
    logging.basicConfig()


#######################################
#     Tests for configure_logging     #
#######################################


@colorlog_available
def test_configure_logging() -> None:
    configure_logging()


def test_configure_logging_without_colorlog() -> None:
    with patch(f"{MODULE}.is_colorlog_available", lambda: False):
        configure_logging()


@pytest.mark.parametrize("level", [logging.INFO, logging.WARNING, logging.ERROR])
def test_configure_logging_level(level: int) -> None:
    with patch(f"{MODULE}.logging.basicConfig") as bc:
        configure_logging(level)
        assert bc.call_args.kwargs["level"] == level


#####################################
#     Tests for log_dict_pretty     #
#####################################


@rich_available
def test_log_dict_pretty_with_rich() -> None:
    with patch(f"{MODULE}.logger") as mock_logger:
        log_dict_pretty({"hello": "world"})

    mock_logger.log.assert_not_called()


@rich_available
def test_log_dict_pretty_with_rich_with_title() -> None:
    with patch(f"{MODULE}.logger") as mock_logger:
        log_dict_pretty({"hello": "world"}, title="cats")

    mock_logger.log.assert_not_called()


def test_log_dict_pretty_with_rich_uses_panel_and_console() -> None:
    data = {"hello": "world"}
    with (
        patch(f"{MODULE}.is_rich_available", return_value=True),
        patch(f"{MODULE}.Console", create=True) as mock_console,
        patch(f"{MODULE}.Pretty", create=True) as mock_pretty,
        patch(f"{MODULE}.Panel", create=True) as mock_panel,
        patch(f"{MODULE}.logger") as mock_logger,
    ):
        console = mock_console.return_value
        panel = mock_panel.return_value
        log_dict_pretty(data, title="cats")

    mock_pretty.assert_called_once_with(data)
    mock_panel.assert_called_once_with(mock_pretty.return_value, title="cats")
    console.print.assert_called_once_with(panel)
    mock_logger.log.assert_not_called()


def test_log_dict_pretty_without_rich() -> None:
    with (
        patch(f"{MODULE}.is_rich_available", return_value=False),
        patch(f"{MODULE}.logger") as mock_logger,
    ):
        log_dict_pretty({"hello": "world"})

    mock_logger.log.assert_called_once_with(logging.INFO, {"hello": "world"})


def test_log_dict_pretty_with_title_without_rich() -> None:
    with (
        patch(f"{MODULE}.is_rich_available", return_value=False),
        patch(f"{MODULE}.logger") as mock_logger,
    ):
        log_dict_pretty({"hello": "world"}, title="cats")

    mock_logger.log.assert_called_once_with(logging.INFO, "cats:\n{'hello': 'world'}")


def test_log_dict_pretty_passes_custom_level_to_logger() -> None:
    with (
        patch(f"{MODULE}.is_rich_available", return_value=False),
        patch(f"{MODULE}.logger") as mock_logger,
    ):
        log_dict_pretty({"hello": "world"}, level=logging.WARNING)

    mock_logger.log.assert_called_once_with(logging.WARNING, {"hello": "world"})
