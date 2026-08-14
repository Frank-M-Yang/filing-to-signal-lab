"""Safety checks for the exact normal Git release candidate."""

from scripts.audit_public_release import audit_public_release


def test_public_release_candidate_has_no_detected_leaks() -> None:
    audit = audit_public_release()

    assert audit.files >= 100
    assert audit.bytes < 5 * 1024 * 1024
    assert audit.issues == ()
