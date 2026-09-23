"""Fail when the Git release candidate contains common privacy or secret leaks."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAX_PUBLIC_FILE_BYTES = 5 * 1024 * 1024
# Source/docs/tests have grown past 5 MiB in aggregate. Keep a separate bounded
# package budget; do not weaken per-file, private-path, or content leak checks.
MAX_PUBLIC_TOTAL_BYTES = 6 * 1024 * 1024
PRIVATE_DATA_ROOTS = (
    "data/interim/",
    "data/market/",
    "data/processed/",
    "data/raw/",
    "data/sec/",
    "output/",
    "outputs/",
)
FORBIDDEN_SUFFIXES = {".key", ".p12", ".pfx", ".pem"}
ALLOWED_EMAIL_DOMAINS = {
    "domain.com",
    "example.com",
    "example.org",
    "massive.com",
    "university.edu",
    "users.noreply.github.com",
}
EMAIL_PATTERN = re.compile(r"[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})", re.IGNORECASE)
# Business Wire's numeric story URL contains a route named "home", not a user
# directory. Exempt only this complete, query-free public URL from the path rule;
# all secret/email checks still inspect the original text.
PUBLIC_NEWSWIRE_STORY_URL = re.compile(
    r"https://www\.businesswire\.com/news/" r"home/[0-9]{14}/en/(?=[\s\"'<>()]|$)"
)
CONTENT_PATTERNS = {
    "Windows absolute path": re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:\\(?!Windows\\Fonts\\)"),
    "macOS/Linux user path": re.compile(r"/(?:Users|home)/[^/\s]+/"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    "OpenAI-style token": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "bearer token": re.compile(r"Bearer [A-Za-z0-9._~+/-]{20,}"),
}


@dataclass(frozen=True)
class ReleaseAudit:
    """Summary of one public candidate audit."""

    files: int
    bytes: int
    issues: tuple[str, ...]


def audit_public_release(repository_root: Path = REPOSITORY_ROOT) -> ReleaseAudit:
    """Inspect exactly the tracked and non-ignored untracked Git candidate files."""
    candidates = _git_candidates(repository_root)
    issues: list[str] = []
    total_bytes = 0
    for relative_path in candidates:
        normalized = relative_path.as_posix()
        full_path = repository_root / relative_path
        if not full_path.is_file():
            continue
        size = full_path.stat().st_size
        total_bytes += size
        if size > MAX_PUBLIC_FILE_BYTES:
            issues.append(f"oversized public file: {normalized} ({size} bytes)")
        if relative_path.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.append(f"credential-like file extension: {normalized}")
        if normalized == ".env" or (
            normalized.startswith(".env.") and normalized != ".env.example"
        ):
            issues.append(f"environment file is public: {normalized}")
        if any(normalized.startswith(root) for root in PRIVATE_DATA_ROOTS):
            if relative_path.name != ".gitkeep":
                issues.append(f"private data/output path is public: {normalized}")

        payload = full_path.read_bytes()
        if b"\x00" in payload:
            continue
        text = payload.decode("utf-8", errors="replace")
        for label, pattern in CONTENT_PATTERNS.items():
            searchable = (
                PUBLIC_NEWSWIRE_STORY_URL.sub("", text)
                if label == "macOS/Linux user path"
                else text
            )
            if pattern.search(searchable):
                issues.append(f"{label}: {normalized}")
        for match in EMAIL_PATTERN.finditer(text):
            domain = match.group(1).lower()
            if domain not in ALLOWED_EMAIL_DOMAINS:
                issues.append(f"non-placeholder email domain {domain}: {normalized}")

    if total_bytes > MAX_PUBLIC_TOTAL_BYTES:
        issues.append(f"oversized public package: {total_bytes} bytes")
    return ReleaseAudit(files=len(candidates), bytes=total_bytes, issues=tuple(sorted(set(issues))))


def _git_candidates(repository_root: Path) -> list[Path]:
    """List files that a normal `git add .` would make public."""
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    return [Path(item.decode("utf-8")) for item in completed.stdout.split(b"\x00") if item]


def main() -> None:
    """Run the release audit and expose a CI-friendly exit status."""
    audit = audit_public_release()
    if audit.issues:
        details = "\n".join(f"- {issue}" for issue in audit.issues)
        raise RuntimeError(f"PUBLIC RELEASE AUDIT FAILED\n{details}")
    print(f"PUBLIC RELEASE AUDIT PASS files={audit.files} bytes={audit.bytes}")


if __name__ == "__main__":
    main()
