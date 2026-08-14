"""End-to-end fixture-backed SEC ingestion and persistence test."""

from datetime import datetime, timezone
from pathlib import Path

from sec_data.models import SecResponse
from sec_data.pipeline import SecIngestionService

FIXTURES = Path(__file__).parent / "fixtures"


class FixtureClient:
    def resolve_ticker(self, ticker: str):
        return (
            "0000000001",
            ticker.upper(),
            "Real Data Demo Corp",
            self._response("ticker-map", b'{"0":{"cik_str":1,"ticker":"RDD"}}'),
        )

    def get_submissions(self, cik: str):
        del cik
        return self._response("submissions", (FIXTURES / "sec_submissions.json").read_bytes())

    def get_companyfacts(self, cik: str):
        del cik
        return self._response("companyfacts", (FIXTURES / "sec_companyfacts.json").read_bytes())

    def get_submission_archive(self, filename: str):
        raise AssertionError(f"Unexpected archive request: {filename}")

    @staticmethod
    def _response(name: str, content: bytes) -> SecResponse:
        return SecResponse(
            url=f"fixture://{name}.json",
            content=content,
            fetched_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            status_code=200,
            headers={"content-type": "application/json"},
        )


def test_pipeline_preserves_raw_and_exports_cleaned_tables(tmp_path: Path) -> None:
    service = SecIngestionService(
        "unused because client is injected",
        data_root=tmp_path / "sec",
        client=FixtureClient(),  # type: ignore[arg-type]
    )

    result = service.ingest("RDD")

    assert result.cik == "0000000001"
    assert len(result.filings) == 2
    assert len(result.facts) == 7
    assert len(result.amendment_comparisons) == 4
    assert len(result.rejections) == 2
    assert len(result.snapshots) == 3
    assert all(snapshot.path.exists() for snapshot in result.snapshots)
    normalized = Path(result.metadata["normalized_directory"])
    assert {path.name for path in normalized.iterdir()} == {
        "filings.csv",
        "facts.csv",
        "amendment_comparisons.csv",
        "rejections.csv",
        "manifest.json",
    }
