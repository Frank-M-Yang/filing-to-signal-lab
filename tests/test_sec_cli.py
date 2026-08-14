"""Command-line validation for the SEC ingestion entry point."""

import pytest

from sec_data.cli import main


def test_cli_explains_missing_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)

    with pytest.raises(SystemExit, match="2"):
        main(["AAPL"])
