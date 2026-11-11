"""The only component that touches the network.

Everything about it is deliberately narrow:

* **An allowlist, checked per hop.** A host not in ``fetcher.allowed_hosts`` is never
  requested, and a redirect that leaves the list is dropped rather than followed —
  which is the hole a check-once-then-follow-redirects client leaves open.
* **A byte ceiling, enforced while streaming.** The limit is not "check
  Content-Length", because a server may lie or omit it; the response is read in chunks
  and abandoned when it goes over.
* **A cache on disk with the extracted text**, so studying the same page twice costs
  nothing and an offline night still has material.
* **Nothing is executed.** Pages become text. Code blocks found in them are data for
  the sandbox to run under its own limits, never something this module runs.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from gustav.fetcher import text as textlib
from gustav.fetcher.sources import Candidate

MAX_REDIRECTS = 3
_SAFE = re.compile(r"[^a-z0-9._-]+")


class FetchError(RuntimeError):
    pass


class NotAllowed(FetchError):
    def __init__(self, url: str, host: str):
        super().__init__(
            f"{host or url!r} is not in fetcher.allowed_hosts, so it was not requested"
        )
        self.url = url
        self.host = host


class TooLarge(FetchError):
    pass


@dataclass
class Fetched:
    url: str
    title: str
    text: str
    sha256: str
    size: int
    path: Path
    content_type: str = "text/plain"
    from_cache: bool = False
    headings: tuple[str, ...] = ()
    code_blocks: tuple[str, ...] = ()


def host_allowed(host: str, allowed: tuple[str, ...] | list[str]) -> bool:
    host = (host or "").lower().rstrip(".")
    if not host:
        return False
    for entry in allowed:
        entry = entry.lower().lstrip("*.").rstrip(".")
        if host == entry or host.endswith("." + entry):
            return True
    return False


class DocFetcher:
    def __init__(
        self,
        *,
        cache_dir: Path,
        allowed_hosts: list[str],
        user_agent: str = "gustav-docs/0.1",
        timeout: float = 20.0,
        max_bytes: int = 2 * 1024 * 1024,
        delay_seconds: float = 1.0,
        network: bool = True,
        sleeper=time.sleep,
    ):
        self.cache_dir = Path(cache_dir)
        self.allowed_hosts = list(allowed_hosts)
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.delay_seconds = delay_seconds
        self.network = network
        self._sleep = sleeper
        self._last_request: dict[str, float] = {}

    # --- cache ----------------------------------------------------------
    def cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
        host = _SAFE.sub("-", (urlparse(url).hostname or "local").lower())
        return self.cache_dir / host / f"{digest}.md"

    def cached(self, url: str) -> Fetched | None:
        path = self.cache_path(url)
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8", errors="replace")
        title, _, body = raw.partition("\n")
        return Fetched(
            url=url,
            title=title.removeprefix("# ").strip(),
            text=body.strip(),
            sha256=hashlib.sha256(body.strip().encode("utf-8")).hexdigest(),
            size=len(body.encode("utf-8")),
            path=path,
            from_cache=True,
        )

    def _store(self, url: str, doc: textlib.Document, content_type: str) -> Fetched:
        path = self.cache_path(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = doc.text.strip()
        path.write_text(f"# {doc.title}\n{body}\n", encoding="utf-8")
        return Fetched(
            url=url,
            title=doc.title,
            text=body,
            sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
            size=len(body.encode("utf-8")),
            path=path,
            content_type=content_type,
            headings=tuple(doc.headings),
            code_blocks=tuple(doc.code_blocks),
        )

    # --- fetching -------------------------------------------------------
    def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme == "file":
            return True
        if parsed.scheme not in ("http", "https"):
            return False
        return host_allowed(parsed.hostname or "", self.allowed_hosts)

    def fetch(self, candidate: Candidate | str, *, refresh: bool = False) -> Fetched:
        url = candidate if isinstance(candidate, str) else candidate.url
        path = candidate.path if isinstance(candidate, Candidate) else None
        if path is None and urlparse(url).scheme == "file":
            # A `file://` entry in docs/sources.yaml is local material named by URL
            # rather than found by scanning: read it, never request it.
            path = Path(url2pathname(urlparse(url).path))

        if path is not None:
            return self._read_local(url, path)
        if not refresh and (hit := self.cached(url)) is not None:
            return hit
        if not self.allowed(url):
            raise NotAllowed(url, urlparse(url).hostname or "")
        if not self.network:
            raise FetchError(
                f"{url} is not cached and the fetcher is disabled (fetcher.enabled: false)"
            )
        body, content_type = self._get(url)
        doc = textlib.read(body, title=urlparse(url).path.rsplit("/", 1)[-1])
        if not doc.text.strip():
            raise FetchError(f"{url} had no readable text")
        return self._store(url, doc, content_type)

    def _read_local(self, url: str, path: Path) -> Fetched:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise FetchError(f"{path}: {exc}") from exc
        if len(raw.encode("utf-8")) > self.max_bytes:
            raise TooLarge(f"{path} is larger than max_bytes_per_document")
        doc = textlib.read(raw, title=path.stem)
        return self._store(url, doc, "text/markdown")

    def _throttle(self, host: str) -> None:
        if self.delay_seconds <= 0:
            return
        last = self._last_request.get(host)
        now = time.monotonic()
        if last is not None and now - last < self.delay_seconds:
            self._sleep(self.delay_seconds - (now - last))
        self._last_request[host] = time.monotonic()

    def _get(self, url: str) -> tuple[str, str]:
        import httpx

        seen: set[str] = set()
        with httpx.Client(
            timeout=self.timeout,
            follow_redirects=False,
            headers={"User-Agent": self.user_agent, "Accept": "text/html,text/plain"},
        ) as client:
            for _hop in range(MAX_REDIRECTS + 1):
                if url in seen:
                    raise FetchError(f"redirect loop at {url}")
                seen.add(url)
                if not self.allowed(url):
                    raise NotAllowed(url, urlparse(url).hostname or "")
                self._throttle(urlparse(url).hostname or "")
                try:
                    with client.stream("GET", url) as response:
                        if response.is_redirect:
                            location = response.headers.get("location")
                            if not location:
                                raise FetchError(f"{url} redirected with no location")
                            url = str(response.url.join(location))
                            continue
                        if response.status_code >= 400:
                            raise FetchError(f"{url} returned HTTP {response.status_code}")
                        chunks: list[bytes] = []
                        total = 0
                        for chunk in response.iter_bytes():
                            total += len(chunk)
                            if total > self.max_bytes:
                                raise TooLarge(
                                    f"{url} is larger than "
                                    f"{self.max_bytes // 1024} KiB and was abandoned"
                                )
                            chunks.append(chunk)
                        content_type = response.headers.get("content-type", "text/html")
                        return b"".join(chunks).decode("utf-8", errors="replace"), content_type
                except httpx.HTTPError as exc:
                    raise FetchError(f"{url}: {type(exc).__name__}: {exc}") from exc
        raise FetchError(f"{url}: too many redirects")
