"""The documentation fetcher: the only component that touches the network.

Nothing here makes a network request. The allowlist, the byte ceiling and the redirect
policy are tested against a stub transport, and everything else runs off local files —
which is also how the fetcher behaves on a machine with no connection.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gustav.fetcher.client import DocFetcher, FetchError, NotAllowed, TooLarge, host_allowed
from gustav.fetcher.sources import load_sources
from gustav.fetcher.text import read, to_text
from gustav.learn.chunk import best_chunks, chunks


# --- the allowlist ------------------------------------------------------
def test_subdomains_are_allowed_but_lookalikes_are_not():
    allowed = ["docs.python.org", "docs.rs"]
    assert host_allowed("docs.python.org", allowed)
    assert host_allowed("legacy.docs.rs", allowed)
    assert not host_allowed("docs.rs.evil.example", allowed)
    assert not host_allowed("notdocs.rs", allowed)
    assert not host_allowed("", allowed)


def test_a_host_outside_the_allowlist_is_never_requested(tmp_path):
    fetcher = DocFetcher(cache_dir=tmp_path, allowed_hosts=["docs.python.org"])
    with pytest.raises(NotAllowed):
        fetcher.fetch("https://example.com/whatever")


def test_non_http_schemes_are_refused(tmp_path):
    fetcher = DocFetcher(cache_dir=tmp_path, allowed_hosts=["docs.python.org"])
    assert not fetcher.allowed("ftp://docs.python.org/x")
    assert not fetcher.allowed("javascript:alert(1)")


def test_with_the_network_off_an_uncached_url_fails_rather_than_being_fetched(tmp_path):
    fetcher = DocFetcher(
        cache_dir=tmp_path, allowed_hosts=["docs.python.org"], network=False
    )
    with pytest.raises(FetchError, match="fetcher is disabled"):
        fetcher.fetch("https://docs.python.org/3/library/re.html")


def test_a_redirect_off_the_allowlist_is_dropped_not_followed(tmp_path, monkeypatch):
    """The hole a check-once client leaves open: the first hop is fine, the second is not."""
    hops: list[str] = []

    class Response:
        def __init__(self, url: str):
            self.url = _Url(url)
            hops.append(url)
            self.is_redirect = "start" in url
            self.status_code = 302 if self.is_redirect else 200
            self.headers = (
                {"location": "https://evil.example/payload"}
                if self.is_redirect
                else {"content-type": "text/html"}
            )

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def iter_bytes(self):
            yield b"<html><body><p>hello</p></body></html>"

    class _Url:
        def __init__(self, url: str):
            self._url = url

        def join(self, other: str) -> str:
            return other

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def stream(self, _method, url):
            return Response(url)

    import httpx

    monkeypatch.setattr(httpx, "Client", Client)
    fetcher = DocFetcher(
        cache_dir=tmp_path, allowed_hosts=["docs.python.org"], delay_seconds=0
    )
    with pytest.raises(NotAllowed):
        fetcher.fetch("https://docs.python.org/start")
    assert hops == ["https://docs.python.org/start"], "the second hop was never requested"


def test_a_document_over_the_ceiling_is_abandoned_mid_stream(tmp_path, monkeypatch):
    class Response:
        url = None
        is_redirect = False
        status_code = 200
        headers = {"content-type": "text/plain"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def iter_bytes(self):
            # No content-length at all: the ceiling has to be enforced while reading.
            for _ in range(100):
                yield b"x" * 1024

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def stream(self, _method, _url):
            return Response()

    import httpx

    monkeypatch.setattr(httpx, "Client", Client)
    fetcher = DocFetcher(
        cache_dir=tmp_path, allowed_hosts=["docs.python.org"], max_bytes=4096,
        delay_seconds=0,
    )
    with pytest.raises(TooLarge):
        fetcher.fetch("https://docs.python.org/big")


# --- local files and the cache -----------------------------------------
def test_a_local_source_is_read_without_the_network(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "lifetime.md").write_text(
        "# Lifetimes\n\nEvery reference has a lifetime.\n", encoding="utf-8"
    )
    sources = load_sources(None, local_dirs=[notes])
    candidates = sources.for_topic("rust", "lifetime")
    assert candidates and candidates[0].local

    fetcher = DocFetcher(cache_dir=tmp_path / "cache", allowed_hosts=[], network=False)
    fetched = fetcher.fetch(candidates[0])
    assert "Every reference" in fetched.text
    assert fetched.path.exists()
    # Second time: the cache answers, with no source file access needed.
    assert fetcher.cached(candidates[0].url) is not None


def test_sources_prefer_the_topics_entry_over_the_module_fallback(tmp_path):
    (tmp_path / "sources.yaml").write_text(
        "sources:\n"
        "  - module: rust\n    urls: [https://doc.rust-lang.org/fallback]\n"
        "  - module: rust\n    topics: [lifetime]\n"
        "    urls: [https://doc.rust-lang.org/lifetimes]\n",
        encoding="utf-8",
    )
    sources = load_sources(tmp_path / "sources.yaml")
    urls = [c.url for c in sources.for_topic("rust", "lifetime")]
    assert urls[0].endswith("lifetimes")
    assert len(urls) == 2
    assert sources.for_topic("coding", "lifetime") == []


def test_the_budget_is_shared_between_topics(tmp_path):
    (tmp_path / "sources.yaml").write_text(
        "sources:\n"
        "  - module: rust\n    topics: [lifetime]\n    urls: [u1, u2, u3]\n"
        "  - module: coding\n    topics: [regex]\n    urls: [v1, v2, v3]\n",
        encoding="utf-8",
    )
    sources = load_sources(tmp_path / "sources.yaml")
    picked = sources.candidates([("rust", "lifetime"), ("coding", "regex")], limit=4)
    assert len({c.topic for c in picked}) == 2, "one topic must not eat the whole budget"
    assert len(picked) == 4


def test_the_shipped_source_map_is_loadable_and_on_the_allowlist(repo_root):
    from gustav.config.schema import Config

    sources = load_sources(repo_root / "docs" / "sources.yaml")
    assert sources.entries and not sources.warnings
    allowed = Config().fetcher.allowed_hosts
    from urllib.parse import urlparse

    for entry in sources.entries:
        for url in entry.urls:
            host = urlparse(url).hostname or ""
            assert host_allowed(host, allowed), f"{host} is not in the default allowlist"


# --- html to text -------------------------------------------------------
def test_navigation_is_dropped_and_code_is_kept():
    doc = to_text(
        "<html><head><title>Re</title></head><body><nav>menu</nav>"
        "<h2>Syntax</h2><p>Use <code>re.compile</code>.</p>"
        "<pre>re.compile('a+')</pre><script>bad()</script></body></html>"
    )
    assert doc.title == "Re"
    assert "menu" not in doc.text
    assert "bad()" not in doc.text
    assert doc.headings == ["Syntax"]
    assert doc.code_blocks == ["re.compile('a+')"]


def test_markdown_is_read_as_markdown():
    doc = read("# Title\n\nbody text\n\n```\ncode()\n```\n")
    assert doc.headings == ["Title"]
    assert doc.code_blocks == ["code()"]


def test_malformed_html_does_not_raise():
    assert read("<html><body><p>half open").text.strip() == "half open"


# --- chunking -----------------------------------------------------------
def test_chunks_follow_headings_and_keep_code_whole():
    text = (
        "# Doc\n\n## First\n\n" + ("Sentence about references. " * 10) + "\n\n"
        "```\nfn main() {}\n```\n\n## Second\n\n" + ("Other prose here. " * 12)
    )
    pieces = chunks(text)
    assert [c.heading for c in pieces] == ["First", "Second"]
    assert "fn main() {}" in pieces[0].text
    assert pieces[0].citation("http://x").endswith("#first")


def test_tiny_sections_are_dropped_unless_they_hold_code():
    pieces = chunks("## Tiny\n\nshort\n\n## Code\n\n```\nx = 1\n```\n")
    assert [c.heading for c in pieces] == ["Code"]


def test_best_chunks_prefers_the_topic_but_keeps_document_order():
    text = (
        "## Alpha\n\n" + ("unrelated prose here. " * 12) + "\n\n"
        "## Lifetime rules\n\n" + ("lifetime lifetime prose. " * 12)
    )
    picked = best_chunks(chunks(text), "lifetime", limit=1)
    assert picked[0].heading == "Lifetime rules"
    both = best_chunks(chunks(text), "lifetime", limit=2)
    assert [c.position for c in both] == sorted(c.position for c in both)


def test_an_unreadable_local_file_is_an_error_not_a_crash(tmp_path):
    fetcher = DocFetcher(cache_dir=tmp_path / "c", allowed_hosts=[], network=False)
    from gustav.fetcher.sources import Candidate

    missing = Path(tmp_path / "nope.md")
    with pytest.raises(FetchError):
        fetcher.fetch(Candidate(url=missing.as_uri(), module="m", topic="t", path=missing))
