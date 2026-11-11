"""Fetch the documentation tonight's topics need.

The only step that uses the network, and the only one that can fail without costing
anything: a page that will not load means one less source to study, not a failed night.

It is polite by construction — one request per host per `fetcher.delay_seconds`, a
cache that is consulted before anything is requested, and a hard ceiling on how many
documents a single night may pull. And it is honest about being offline: with
`fetcher.enabled: false` it studies `fetcher.local_sources` and whatever is already
cached, and says so in the result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from gustav.fetcher.client import DocFetcher, FetchError
from gustav.fetcher.sources import load_sources
from gustav.learn.docs import DocumentStore
from gustav.sleep.stages._base import StepEnv, sleep_step


@sleep_step
def run_step(env: StepEnv) -> dict[str, Any]:
    cfg = env.config.fetcher
    topics_payload = env.read_step_json("reflect", "topics.json", {}) or {}
    topics = [(t["module"], t["topic"]) for t in topics_payload.get("topics", [])]

    # A relative `local_sources` entry means "in the project", not "wherever this
    # child process happens to have been started".
    local_dirs = [
        directory if (directory := Path(entry).expanduser()).is_absolute()
        else env.paths.root / directory
        for entry in cfg.local_sources
    ]
    sources = load_sources(env.paths.root / "docs" / "sources.yaml", local_dirs=local_dirs)
    for warning in sources.warnings:
        env.log(warning, level="warning")

    result: dict[str, Any] = {
        "topics": [f"{m}/{t}" for m, t in topics],
        "network": cfg.enabled,
        "fetched": 0,
        "cached": 0,
        "failed": 0,
        "refused": 0,
        "unlicensed": 0,
        "documents": [],
    }
    if not topics:
        result["reason"] = "reflect chose no topics"
        return result

    candidates = sources.candidates(topics, limit=cfg.max_documents_per_run)
    if not candidates:
        result["reason"] = "no source in docs/sources.yaml matches tonight's topics"
        env.log(result["reason"], level="warning")
        return result
    if env.dry_run:
        result["dry_run"] = True
        result["documents"] = [{"url": c.url, "topic": c.topic} for c in candidates]
        return result

    docs = DocumentStore(env.db)
    fetcher = DocFetcher(
        cache_dir=env.paths.doc_cache,
        allowed_hosts=cfg.allowed_hosts,
        user_agent=cfg.user_agent,
        timeout=cfg.timeout_seconds,
        max_bytes=cfg.max_bytes_per_document,
        delay_seconds=cfg.delay_seconds,
        network=cfg.enabled,
    )

    for _index, candidate in env.each(candidates, label="document", checkpoint_key="doc_index"):
        age = docs.age_hours(candidate.url)
        fresh_enough = age is not None and age < cfg.cache_ttl_hours
        if fresh_enough and fetcher.cached(candidate.url) is not None:
            result["cached"] += 1
            continue
        try:
            fetched = fetcher.fetch(candidate)
        except FetchError as exc:
            kind = "refused" if "allowed_hosts" in str(exc) else "failed"
            result[kind] += 1
            docs.record(
                url=candidate.url, topic=candidate.topic, title="", sha256="", size=0,
                path="", host=candidate.host, status="failed", run_id=env.run_id,
                error=str(exc)[:400],
            )
            env.log(f"{candidate.url}: {exc}", level="warning")
            continue

        if not candidate.license:
            # Not fatal, and not silent: training on documentation is a licence
            # question, and the answer has to be recorded where you can find it.
            result["unlicensed"] += 1
            env.log(
                f"no licence recorded for {candidate.url} in docs/sources.yaml; "
                f"whether this may be trained on is your call",
                level="warning",
            )
        docs.record(
            url=candidate.url,
            topic=candidate.topic,
            title=fetched.title,
            sha256=fetched.sha256,
            size=fetched.size,
            path=str(fetched.path),
            host=candidate.host or urlparse(candidate.url).scheme,
            content_type=fetched.content_type,
            license=candidate.license,
            status="cached" if fetched.from_cache else "fresh",
            run_id=env.run_id,
        )
        result["cached" if fetched.from_cache else "fetched"] += 1
        result["documents"].append(
            {
                "url": candidate.url,
                "topic": candidate.topic,
                "title": fetched.title,
                "kib": round(fetched.size / 1024, 1),
                "license": candidate.license,
                "local": candidate.local,
            }
        )

    env.write_json("documents.json", result)
    env.events.emit(
        "sleep.docs",
        f"{result['fetched']} documents fetched, {result['cached']} already cached, "
        f"{result['failed'] + result['refused']} unavailable",
        run_id=env.run_id,
        **{k: result[k] for k in ("fetched", "cached", "failed", "refused")},
    )
    return result
