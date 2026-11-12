"""Where tools live: on disk so you can read them, in SQLite so the daemon can find them.

``tools/<name>/vNNN/tool.py`` with a ``meta.json`` beside it, and an ``active`` symlink
pointing at the version in use — the same shape as the adapters, for the same reason.
A version is never overwritten: a tool the model rewrites is a new version that has to
be reviewed on its own, and the one you approved keeps working until you say otherwise.

Nothing here decides whether a tool may run. That is the ``status`` column, and it only
reaches ``approved`` because a human put it there.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gustav.store.db import Database
from gustav.tools.review import Checks
from gustav.tools.spec import ToolSpec

VERSION_RE = re.compile(r"^v(\d{3,})$")
ACTIVE = "active"


class ToolError(RuntimeError):
    pass


@dataclass
class ToolVersion:
    name: str
    version: str
    source: str
    parameters: dict[str, Any] = field(default_factory=dict)
    capabilities: tuple[str, ...] = ("pure",)
    examples: tuple[dict[str, Any], ...] = ()
    checks: dict[str, Any] = field(default_factory=dict)
    status: str = "proposed"
    author: str = "model"
    created_at: str = ""
    reviewed_at: str | None = None
    reviewed_by: str | None = None
    note: str | None = None
    description: str = ""
    path: Path | None = None

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            source=self.source,
            parameters=self.parameters,
            capabilities=tuple(self.capabilities),
            examples=tuple(self.examples),
            author=self.author,
        )

    def describe(self) -> str:
        return f"{self.name}@{self.version} ({self.status})"


@dataclass
class ToolRecord:
    name: str
    description: str
    status: str
    active_version: str | None
    created_at: str = ""
    updated_at: str = ""
    versions: int = 0

    @property
    def callable(self) -> bool:
        return self.status == "approved" and bool(self.active_version)


class ToolStore:
    def __init__(self, db: Database, root: Path | str, *, max_tools: int = 20):
        self.db = db
        self.root = Path(root)
        self.max_tools = max_tools

    # --- disk -----------------------------------------------------------
    def tool_dir(self, name: str) -> Path:
        return self.root / name

    def version_dir(self, name: str, version: str) -> Path:
        return self.tool_dir(name) / version

    def next_version(self, name: str) -> str:
        directory = self.tool_dir(name)
        numbers = [
            int(match.group(1))
            for child in (directory.iterdir() if directory.exists() else [])
            if child.is_dir() and not child.is_symlink() and (match := VERSION_RE.match(child.name))
        ]
        return f"v{(max(numbers) + 1) if numbers else 1:03d}"

    def _write_files(self, version: ToolVersion) -> Path:
        directory = self.version_dir(version.name, version.version)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tool.py").write_text(version.source, encoding="utf-8")
        meta = {
            "name": version.name,
            "version": version.version,
            "description": version.description,
            "parameters": version.parameters,
            "capabilities": list(version.capabilities),
            "examples": list(version.examples),
            "checks": version.checks,
            "status": version.status,
            "author": version.author,
            "created_at": version.created_at,
        }
        temporary = directory / ".meta.json.part"
        temporary.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        os.replace(temporary, directory / "meta.json")
        return directory

    def _activate_link(self, name: str, version: str) -> None:
        link = self.tool_dir(name) / ACTIVE
        temporary = self.tool_dir(name) / f".{ACTIVE}.swap"
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(version)
        os.replace(temporary, link)

    # --- proposing ------------------------------------------------------
    def propose(
        self,
        spec: ToolSpec,
        checks: Checks,
        *,
        run_id: int | None = None,
        request_id: int | None = None,
        author: str = "model",
    ) -> ToolVersion:
        """Record a new version of a tool, awaiting review."""
        row = self.db.query_one("SELECT * FROM tools WHERE name = ?", (spec.name,))
        if row is None:
            if self.count() >= self.max_tools:
                raise ToolError(
                    f"there are already {self.count()} tools and tools.max_tools is "
                    f"{self.max_tools}"
                )
            tool_id = self.db.insert(
                "tools",
                {
                    "name": spec.name,
                    "description": spec.description,
                    "status": "proposed",
                    "request_id": request_id,
                    "run_id": run_id,
                },
            )
        else:
            tool_id = int(row["id"])
            self.db.execute(
                "UPDATE tools SET description = ?, updated_at = datetime('now')"
                " WHERE id = ?",
                (spec.description, tool_id),
            )

        version = self.next_version(spec.name)
        record = ToolVersion(
            name=spec.name,
            version=version,
            source=spec.source,
            parameters=spec.parameters,
            capabilities=tuple(spec.capabilities),
            examples=tuple(spec.examples),
            checks=checks.to_dict(),
            status="proposed",
            author=author,
            description=spec.description,
        )
        try:
            self.db.insert(
                "tool_versions",
                {
                    "tool_id": tool_id,
                    "version": version,
                    "source": spec.source,
                    "parameters": json.dumps(spec.parameters),
                    "capabilities": json.dumps(list(spec.capabilities)),
                    "examples": json.dumps(list(spec.examples)),
                    "checks": json.dumps(checks.to_dict(), default=str),
                    "status": "proposed",
                    "author": author,
                    "run_id": run_id,
                    "fingerprint": spec.fingerprint,
                },
            )
        except Exception as exc:
            if "UNIQUE" in str(exc):
                raise ToolError(
                    f"this exact version of {spec.name!r} has been proposed before"
                ) from exc
            raise
        record.path = self._write_files(record)
        if request_id is not None:
            self.db.execute(
                "UPDATE tool_requests SET status = 'written', tool_id = ?,"
                " detail = ? WHERE id = ?",
                (tool_id, f"{spec.name}@{version} awaits review", request_id),
            )
        return record

    # --- review ---------------------------------------------------------
    def approve(
        self, name: str, version: str | None = None, *, by: str = "user", note: str = ""
    ) -> ToolVersion:
        """Make a version callable. The only path to it, and it needs a human."""
        record = self._version_row(name, version)
        if record["status"] == "approved":
            raise ToolError(f"{name}@{record['version']} is already approved")
        checks = json.loads(record["checks"] or "{}")
        if not checks.get("ok", False):
            raise ToolError(
                f"{name}@{record['version']} did not pass its admission checks, so it "
                f"cannot be approved: " + "; ".join(checks.get("problems") or ["unknown"])
            )
        self.db.execute(
            "UPDATE tool_versions SET status = 'superseded' WHERE tool_id = ?"
            " AND status = 'approved'",
            (record["tool_id"],),
        )
        self.db.execute(
            "UPDATE tool_versions SET status = 'approved', reviewed_at = datetime('now'),"
            " reviewed_by = ?, note = ? WHERE id = ?",
            (by, note or None, record["id"]),
        )
        self.db.execute(
            "UPDATE tools SET status = 'approved', active_version = ?,"
            " updated_at = datetime('now') WHERE id = ?",
            (record["version"], record["tool_id"]),
        )
        self._activate_link(name, record["version"])
        version_record = self._to_version(record)
        version_record.status = "approved"
        version_record.reviewed_by = by
        self._write_files(version_record)
        return version_record

    def reject(
        self, name: str, version: str | None = None, *, by: str = "user", note: str = ""
    ) -> ToolVersion:
        record = self._version_row(name, version)
        self.db.execute(
            "UPDATE tool_versions SET status = 'rejected', reviewed_at = datetime('now'),"
            " reviewed_by = ?, note = ? WHERE id = ?",
            (by, note or None, record["id"]),
        )
        remaining = self.db.query_one(
            "SELECT COUNT(*) AS n FROM tool_versions WHERE tool_id = ?"
            " AND status = 'approved'",
            (record["tool_id"],),
        )
        if not remaining["n"]:
            self.db.execute(
                "UPDATE tools SET status = 'rejected', active_version = NULL,"
                " updated_at = datetime('now') WHERE id = ?",
                (record["tool_id"],),
            )
        version_record = self._to_version(record)
        version_record.status = "rejected"
        version_record.note = note
        self._write_files(version_record)
        return version_record

    def retire(self, name: str, *, note: str = "") -> None:
        """Stop offering a tool without deleting anything."""
        row = self._tool_row(name)
        self.db.execute(
            "UPDATE tools SET status = 'retired', active_version = NULL,"
            " updated_at = datetime('now') WHERE id = ?",
            (row["id"],),
        )
        link = self.tool_dir(name) / ACTIVE
        if link.is_symlink() or link.exists():
            link.unlink()

    def rollback(self, name: str, *, by: str = "user") -> ToolVersion:
        """Go back to the previous approved version of a tool."""
        rows = self.db.query(
            "SELECT v.* FROM tool_versions v JOIN tools t ON t.id = v.tool_id"
            " WHERE t.name = ? AND v.status IN ('approved', 'superseded')"
            " ORDER BY v.id DESC",
            (name,),
        )
        if len(rows) < 2:
            raise ToolError(f"{name} has no earlier approved version to go back to")
        return self.approve(name, rows[1]["version"], by=by, note="rolled back")

    # --- reading --------------------------------------------------------
    def _tool_row(self, name: str):
        row = self.db.query_one("SELECT * FROM tools WHERE name = ?", (name,))
        if row is None:
            raise ToolError(f"there is no tool called {name!r}")
        return row

    def _version_row(self, name: str, version: str | None):
        tool = self._tool_row(name)
        if version is None:
            row = self.db.query_one(
                "SELECT * FROM tool_versions WHERE tool_id = ? ORDER BY id DESC LIMIT 1",
                (tool["id"],),
            )
        else:
            row = self.db.query_one(
                "SELECT * FROM tool_versions WHERE tool_id = ? AND version = ?",
                (tool["id"], version),
            )
        if row is None:
            raise ToolError(f"{name} has no version {version or '(any)'}")
        merged = dict(row)
        merged["description"] = tool["description"]
        return merged

    def _to_version(self, row: Any) -> ToolVersion:
        row = dict(row)
        return ToolVersion(
            name=self.db.scalar(
                "SELECT name FROM tools WHERE id = ?", (row["tool_id"],), default=""
            ),
            version=row["version"],
            source=row["source"],
            parameters=json.loads(row["parameters"] or "{}"),
            capabilities=tuple(json.loads(row["capabilities"] or "[]")),
            examples=tuple(json.loads(row["examples"] or "[]")),
            checks=json.loads(row["checks"] or "{}"),
            status=row["status"],
            author=row["author"],
            created_at=row["created_at"],
            reviewed_at=row["reviewed_at"],
            reviewed_by=row["reviewed_by"],
            note=row["note"],
            description=row.get("description", ""),
            path=self.version_dir(
                self.db.scalar(
                    "SELECT name FROM tools WHERE id = ?", (row["tool_id"],), default=""
                ),
                row["version"],
            ),
        )

    def version(self, name: str, version: str | None = None) -> ToolVersion:
        return self._to_version(self._version_row(name, version))

    def versions(self, name: str) -> list[ToolVersion]:
        tool = self._tool_row(name)
        return [
            self._to_version(row)
            for row in self.db.query(
                "SELECT * FROM tool_versions WHERE tool_id = ? ORDER BY id", (tool["id"],)
            )
        ]

    def record(self, name: str) -> ToolRecord | None:
        row = self.db.query_one(
            "SELECT t.*, (SELECT COUNT(*) FROM tool_versions v WHERE v.tool_id = t.id)"
            " AS versions FROM tools t WHERE t.name = ?",
            (name,),
        )
        return self._to_record(row) if row else None

    @staticmethod
    def _to_record(row: Any) -> ToolRecord:
        return ToolRecord(
            name=row["name"],
            description=row["description"],
            status=row["status"],
            active_version=row["active_version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            versions=int(row["versions"] if "versions" in row.keys() else 0),
        )

    def all(self) -> list[ToolRecord]:
        rows = self.db.query(
            "SELECT t.*, (SELECT COUNT(*) FROM tool_versions v WHERE v.tool_id = t.id)"
            " AS versions FROM tools t ORDER BY t.name"
        )
        return [self._to_record(row) for row in rows]

    def count(self) -> int:
        return int(
            self.db.scalar(
                "SELECT COUNT(*) FROM tools WHERE status != 'retired'", default=0
            )
            or 0
        )

    def approved(self) -> list[ToolVersion]:
        """The tools the model may actually call."""
        rows = self.db.query(
            "SELECT v.* FROM tool_versions v JOIN tools t ON t.id = v.tool_id"
            " WHERE t.status = 'approved' AND v.status = 'approved'"
            " AND v.version = t.active_version ORDER BY t.name"
        )
        out = []
        for row in rows:
            version = self._to_version(row)
            version.description = self.db.scalar(
                "SELECT description FROM tools WHERE id = ?", (row["tool_id"],), default=""
            )
            out.append(version)
        return out

    def pending(self) -> list[ToolVersion]:
        """Versions waiting for a human to read them."""
        rows = self.db.query(
            "SELECT v.* FROM tool_versions v WHERE v.status = 'proposed' ORDER BY v.id"
        )
        out = []
        for row in rows:
            version = self._to_version(row)
            version.description = self.db.scalar(
                "SELECT description FROM tools WHERE id = ?", (row["tool_id"],), default=""
            )
            out.append(version)
        return out

    # --- calls ----------------------------------------------------------
    def record_call(
        self,
        name: str,
        *,
        version: str | None,
        arguments: dict[str, Any],
        result: Any,
        ok: bool,
        error: str = "",
        duration_ms: int = 0,
        session_id: int | None = None,
        turn_id: int | None = None,
        source: str = "chat",
    ) -> int:
        tool_id = self.db.scalar("SELECT id FROM tools WHERE name = ?", (name,))
        return self.db.insert(
            "tool_calls",
            {
                "tool_id": tool_id,
                "name": name,
                "version": version,
                "session_id": session_id,
                "turn_id": turn_id,
                "arguments": json.dumps(arguments, default=str)[:4000],
                "result": json.dumps(result, default=str)[:4000] if ok else None,
                "ok": int(bool(ok)),
                "error": error[:1000] or None,
                "duration_ms": duration_ms,
                "source": source,
            },
        )

    def calls(self, name: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        if name:
            rows = self.db.query(
                "SELECT * FROM tool_calls WHERE name = ? ORDER BY id DESC LIMIT ?",
                (name, limit),
            )
        else:
            rows = self.db.query(
                "SELECT * FROM tool_calls ORDER BY id DESC LIMIT ?", (limit,)
            )
        return [dict(row) for row in rows]

    def usage(self, name: str) -> dict[str, Any]:
        row = self.db.query_one(
            "SELECT COUNT(*) AS calls, SUM(ok) AS ok, MAX(created_at) AS last"
            " FROM tool_calls WHERE name = ?",
            (name,),
        )
        calls = int(row["calls"] or 0)
        return {
            "calls": calls,
            "ok": int(row["ok"] or 0),
            "failed": calls - int(row["ok"] or 0),
            "last": row["last"],
        }

    # --- requests -------------------------------------------------------
    def add_request(self, text: str, *, source: str = "user") -> int:
        return self.db.insert("tool_requests", {"text": text.strip(), "source": source})

    def open_requests(self, limit: int = 20) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.db.query(
                "SELECT * FROM tool_requests WHERE status = 'open' ORDER BY id LIMIT ?",
                (limit,),
            )
        ]

    def close_request(self, request_id: int, *, status: str, detail: str = "") -> None:
        self.db.execute(
            "UPDATE tool_requests SET status = ?, detail = ? WHERE id = ?",
            (status, detail[:500] or None, request_id),
        )

    def withdraw_request(self, request_id: int) -> bool:
        cur = self.db.execute(
            "UPDATE tool_requests SET status = 'withdrawn' WHERE id = ? AND status = 'open'",
            (request_id,),
        )
        return bool(cur.rowcount)

    def delete(self, name: str) -> Path | None:
        """Remove a tool and its versions from disk. The log of its calls is kept."""
        row = self._tool_row(name)
        self.db.execute("DELETE FROM tools WHERE id = ?", (row["id"],))
        directory = self.tool_dir(name)
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
            return directory
        return None
