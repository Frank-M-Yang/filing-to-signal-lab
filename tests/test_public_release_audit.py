"""Safety checks for the exact normal Git release candidate."""

from pathlib import Path

import pytest

from scripts import audit_public_release as release
from scripts.audit_public_release import MAX_PUBLIC_TOTAL_BYTES, audit_public_release


def test_public_release_candidate_has_no_detected_leaks() -> None:
    audit = audit_public_release()

    assert audit.files >= 100
    assert audit.bytes <= MAX_PUBLIC_TOTAL_BYTES
    assert audit.issues == ()


def test_single_file_budget_still_rejects_five_mib_plus_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(release, "_git_candidates", lambda root: [Path("large.bin")])
    (tmp_path / "large.bin").write_bytes(b"\x00" * (release.MAX_PUBLIC_FILE_BYTES + 1))
    audit = audit_public_release(tmp_path)
    assert any(issue.startswith("oversized public file:") for issue in audit.issues)


def test_aggregate_budget_is_enforced_by_the_cli_audit_not_only_this_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [Path("one.bin"), Path("two.bin")]
    monkeypatch.setattr(release, "_git_candidates", lambda root: candidates)
    for path in candidates:
        (tmp_path / path).write_bytes(b"\x00" * (MAX_PUBLIC_TOTAL_BYTES // 2 + 1))
    audit = audit_public_release(tmp_path)
    assert any(issue.startswith("oversized public package:") for issue in audit.issues)
    assert not any(issue.startswith("oversized public file:") for issue in audit.issues)


@pytest.mark.parametrize("case", ["public_story", "real_path", "another_host", "query", "email"])
def test_public_story_route_is_not_a_user_path_but_real_leaks_still_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    url = "https://www.businesswire.com/news/" + "home/20250422312631/en/"
    text = url
    if case == "real_path":
        text += "\n/" + "home/" + "example-person/private.txt"
    elif case == "another_host":
        text = url.replace("www.businesswire.com", "unreviewed.example.org")
    elif case == "query":
        text += "?value=something"
    elif case == "email":
        text += "\nexample-person" + "@" + "personal.invalid"
    monkeypatch.setattr(release, "_git_candidates", lambda root: [Path("doc.md")])
    (tmp_path / "doc.md").write_text(text, encoding="utf-8")
    issues = audit_public_release(tmp_path).issues
    assert not issues if case == "public_story" else bool(issues)
