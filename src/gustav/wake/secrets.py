"""Secret detection for the wake phase.

Two separate jobs, and it is worth being clear about both:

1. **Never train on secrets.** A turn with a detected secret is marked untrainable, so
   the sleep pipeline cannot pick it up. This is what section 6 of the spec asks for.
2. **Never write them to disk either.** The default goes further and stores a redacted
   copy, because the store is backed up, copied between machines and summarised into
   reports. The live request still sees the original text, so the answer is unaffected;
   only what is persisted changes. Set ``wake.redact_secrets: false`` to keep the raw
   text on disk (still untrainable).

Detection is deliberately eager: a false positive costs one turn of training data, a
false negative puts a credential in a file that gets copied around.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PLACEHOLDER = "[REDACTED:{kind}]"


@dataclass(frozen=True)
class Pattern:
    kind: str
    regex: re.Pattern[str]
    #: Which capture group holds the secret itself (0 = the whole match).
    group: int = 0


def _p(kind: str, pattern: str, group: int = 0, flags: int = 0) -> Pattern:
    return Pattern(kind, re.compile(pattern, flags), group)


PATTERNS: tuple[Pattern, ...] = (
    _p("private-key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
       flags=re.DOTALL),
    _p("private-key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]{0,40}"),
    _p("aws-access-key", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    _p("aws-secret-key",
       r"(?i)\baws_secret_access_key\b\s*[:=]\s*[\"']?([A-Za-z0-9/+=]{40})", 1),
    _p("github-token", r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b"),
    _p("github-token", r"\bgithub_pat_[A-Za-z0-9_]{22,}\b"),
    # Anthropic first, and the OpenAI pattern excludes "sk-ant-", so the *kind* we
    # report is right — it is shown to the user, so a wrong label is confusing.
    _p("anthropic-key", r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    _p("openai-key", r"\bsk-(?!ant-)(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    # Google keys are AIza + 35 characters, but being exact here only risks missing one.
    _p("google-api-key", r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    _p("slack-token", r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b"),
    _p("stripe-key", r"\b(?:sk|rk)_live_[0-9A-Za-z]{16,}\b"),
    _p("hugging-face-token", r"\bhf_[A-Za-z0-9]{30,}\b"),
    _p("jwt", r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    _p("url-credentials", r"\b[a-z][a-z0-9+.-]*://[^\s/:@]+:([^\s/@]{3,})@", 1),
    # Generic "name = value" assignments. The value must look deliberate, not prose.
    _p("credential-assignment",
       r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token"
       r"|client[_-]?secret|private[_-]?token|bearer)\b\s*[:=]\s*[\"']?"
       r"([^\s\"',;]{8,})", 1),
    _p("authorization-header", r"(?i)\bauthorization\s*:\s*(?:bearer|basic)\s+([A-Za-z0-9._\-+/=]{12,})", 1),
    _p("env-assignment", r"(?m)^\s*[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)\s*=\s*(\S{8,})$", 1),
)

#: Values that look like secrets but are obviously not.
_IGNORED_VALUES = {
    "changeme", "password", "secret", "your_api_key", "your-api-key", "xxxxxxxx",
    "none", "null", "true", "false", "redacted", "placeholder", "example",
    "<your-key>", "...", "undefined", "env", "os.environ",
}


@dataclass(frozen=True)
class Finding:
    kind: str
    start: int
    end: int
    preview: str

    def describe(self) -> str:
        return f"{self.kind} at {self.start}-{self.end} ({self.preview})"


def _looks_real(value: str) -> bool:
    stripped = value.strip().strip("\"'`")
    if len(stripped) < 8:
        return False
    if stripped.lower() in _IGNORED_VALUES:
        return False
    if stripped.startswith(("$", "{{", "${", "<")):
        return False  # a variable reference or a template placeholder
    if re.fullmatch(r"[xX*.\-_]+", stripped):
        return False  # already masked
    return True


def scan(text: str) -> list[Finding]:
    """Every secret-looking span in ``text``, earliest first, without overlaps."""
    findings: list[Finding] = []
    for pattern in PATTERNS:
        for match in pattern.regex.finditer(text):
            group = pattern.group if pattern.group <= (match.re.groups or 0) else 0
            value = match.group(group)
            if value is None:
                continue
            if pattern.group and not _looks_real(value):
                continue
            start, end = match.span(group)
            findings.append(
                Finding(kind=pattern.kind, start=start, end=end, preview=_preview(value))
            )
    return _dedupe(findings)


def _dedupe(findings: list[Finding]) -> list[Finding]:
    ordered = sorted(findings, key=lambda f: (f.start, -(f.end - f.start)))
    out: list[Finding] = []
    for finding in ordered:
        if out and finding.start < out[-1].end:
            continue  # contained in or overlapping an earlier, longer match
        out.append(finding)
    return out


def _preview(value: str) -> str:
    value = value.strip()
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:3]}...{value[-2:]} ({len(value)} chars)"


def redact(text: str) -> tuple[str, list[Finding]]:
    """Replace every finding with a placeholder. Returns the new text and the findings."""
    findings = scan(text)
    if not findings:
        return text, []
    out: list[str] = []
    cursor = 0
    for finding in findings:
        out.append(text[cursor : finding.start])
        out.append(PLACEHOLDER.format(kind=finding.kind))
        cursor = finding.end
    out.append(text[cursor:])
    return "".join(out), findings


def contains_secret(text: str) -> bool:
    return bool(scan(text))


def summarise(findings: list[Finding]) -> str:
    kinds = sorted({f.kind for f in findings})
    return ", ".join(kinds)
