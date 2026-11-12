"""Accounts, passwords, tokens and sessions.

The daemon has always had exactly one kind of caller: whoever can open the Unix socket,
which the filesystem decides. That is the right model for a personal assistant on your
own machine, and it is kept — a socket caller is the owner, with no password involved.

Turning the TCP API on changes the question, because now the network can ask. So this
adds the smallest thing that answers it honestly:

* **Accounts**, with PBKDF2-HMAC-SHA256 password hashes (600 000 iterations, a random
  salt per password, constant-time comparison). No third-party dependency, no plaintext
  anywhere, and the hash format records its own parameters so the cost can be raised
  later without invalidating what is stored.
* **API tokens** for the CLI: shown once when created, stored only as a SHA-256 hash,
  revocable, with their last use recorded so a forgotten one is visible.
* **Browser sessions** for the WebUI: the same thing with an expiry and an httpOnly
  cookie.
* **Two roles.** ``owner`` can review tools, approve expansions, see the whole system;
  ``user`` can chat and see their own chats. The split exists because approving a tool
  means letting generated code run on this machine — that is not a thing to hand out
  with a chat account.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from gustav.store.db import Database

#: Password hashing. The iteration count is stored with each hash, so raising it here
#: upgrades new passwords without breaking existing ones.
ITERATIONS = 600_000
ALGORITHM = "pbkdf2_sha256"
TOKEN_PREFIX = "gus_"
#: The browser session cookie. httpOnly and SameSite=Lax when it is set.
COOKIE = "gustav_session"
#: Roles, most privileged first.
ROLES = ("owner", "user")


class AuthError(RuntimeError):
    """Something was wrong with an account, a password or a token."""


def hash_password(password: str, *, iterations: int = ITERATIONS) -> str:
    if not password or len(password) < 8:
        raise AuthError("a password needs at least 8 characters")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{ALGORITHM}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(stored: str, password: str) -> bool:
    """Constant-time check. False for anything malformed rather than an exception."""
    try:
        algorithm, iterations, salt_hex, digest_hex = (stored or "").split("$")
        if algorithm != ALGORITHM:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", (password or "").encode("utf-8"), bytes.fromhex(salt_hex),
            int(iterations),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), digest_hex)


def new_token() -> tuple[str, str]:
    """A token and its hash. The token is returned once and never stored."""
    token = TOKEN_PREFIX + secrets.token_urlsafe(32)
    return token, hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_hash(token: str) -> str:
    return hashlib.sha256((token or "").encode("utf-8")).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class User:
    id: int
    name: str
    role: str = "user"
    disabled: bool = False
    created_at: str = ""
    last_seen_at: str | None = None
    note: str | None = None

    @property
    def is_owner(self) -> bool:
        return self.role == "owner"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "disabled": self.disabled,
            "created_at": self.created_at,
            "last_seen_at": self.last_seen_at,
        }


@dataclass
class Identity:
    """Who is asking. The owner-over-the-socket case has no user attached."""

    kind: str  # socket | token | cookie | service
    user: User | None = None
    token_id: int | None = None

    @property
    def is_owner(self) -> bool:
        if self.kind in ("socket", "service"):
            return True
        return bool(self.user and self.user.is_owner)

    @property
    def name(self) -> str:
        if self.user is not None:
            return self.user.name
        return "owner" if self.kind in ("socket", "service") else "anonymous"

    @property
    def user_id(self) -> int | None:
        return self.user.id if self.user else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": "owner" if self.is_owner else "user",
            "via": self.kind,
            "user": self.user.to_dict() if self.user else None,
        }


class Accounts:
    """Users, their tokens and their browser sessions."""

    def __init__(self, db: Database, *, session_days: int = 30):
        self.db = db
        self.session_days = session_days

    # --- users ----------------------------------------------------------
    def create(
        self, name: str, password: str, *, role: str = "user", note: str = ""
    ) -> User:
        name = (name or "").strip().lower()
        if not name or not name.replace("-", "").replace("_", "").isalnum():
            raise AuthError(
                f"{name!r} is not a usable account name: letters, digits, - and _"
            )
        if role not in ROLES:
            raise AuthError(f"unknown role {role!r}; expected one of {', '.join(ROLES)}")
        if self.find(name) is not None:
            raise AuthError(f"there is already an account called {name!r}")
        user_id = self.db.insert(
            "users",
            {
                "name": name,
                "password_hash": hash_password(password),
                "role": role,
                "note": note or None,
            },
        )
        return self.get(user_id)

    def get(self, user_id: int) -> User:
        row = self.db.query_one("SELECT * FROM users WHERE id = ?", (user_id,))
        if row is None:
            raise AuthError(f"there is no account with id {user_id}")
        return self._to_user(row)

    def find(self, name: str) -> User | None:
        row = self.db.query_one(
            "SELECT * FROM users WHERE name = ?", ((name or "").strip().lower(),)
        )
        return self._to_user(row) if row else None

    def all(self) -> list[User]:
        return [
            self._to_user(row)
            for row in self.db.query("SELECT * FROM users ORDER BY id")
        ]

    def count(self) -> int:
        return int(self.db.scalar("SELECT COUNT(*) FROM users", default=0) or 0)

    def set_password(self, name: str, password: str) -> User:
        user = self._require(name)
        self.db.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (hash_password(password), user.id),
        )
        # A new password ends every session and token: that is what changing it is for.
        self.revoke_all(user.id)
        return user

    def set_role(self, name: str, role: str) -> User:
        if role not in ROLES:
            raise AuthError(f"unknown role {role!r}")
        user = self._require(name)
        self.db.execute("UPDATE users SET role = ? WHERE id = ?", (role, user.id))
        return self.get(user.id)

    def disable(self, name: str, *, disabled: bool = True) -> User:
        user = self._require(name)
        self.db.execute(
            "UPDATE users SET disabled = ? WHERE id = ?", (int(disabled), user.id)
        )
        if disabled:
            self.revoke_all(user.id)
        return self.get(user.id)

    def delete(self, name: str) -> None:
        user = self._require(name)
        self.revoke_all(user.id)
        self.db.execute("DELETE FROM users WHERE id = ?", (user.id,))

    def authenticate(self, name: str, password: str) -> User:
        user = self.find(name)
        row = (
            self.db.query_one("SELECT password_hash FROM users WHERE id = ?", (user.id,))
            if user
            else None
        )
        # The same work either way, so a missing account and a wrong password take the
        # same time and look the same from outside.
        stored = row["password_hash"] if row else hash_password("x" * 12)
        if not verify_password(stored, password) or user is None:
            raise AuthError("that account and password do not match")
        if user.disabled:
            raise AuthError(f"the account {user.name!r} is disabled")
        self.db.execute(
            "UPDATE users SET last_seen_at = datetime('now') WHERE id = ?", (user.id,)
        )
        return user

    def _require(self, name: str) -> User:
        user = self.find(name)
        if user is None:
            raise AuthError(f"there is no account called {name!r}")
        return user

    @staticmethod
    def _to_user(row: Any) -> User:
        return User(
            id=int(row["id"]),
            name=row["name"],
            role=row["role"],
            disabled=bool(row["disabled"]),
            created_at=row["created_at"],
            last_seen_at=row["last_seen_at"],
            note=row["note"],
        )

    # --- API tokens -----------------------------------------------------
    def issue_token(self, name: str, *, label: str = "cli", days: int | None = None) -> str:
        """Create a token for an account. Returned once; only its hash is kept."""
        user = self._require(name)
        token, digest = new_token()
        expires = (
            (_now() + timedelta(days=days)).isoformat(timespec="seconds")
            if days
            else None
        )
        self.db.insert(
            "api_tokens",
            {
                "user_id": user.id,
                "name": label,
                "token_hash": digest,
                "expires_at": expires,
            },
        )
        return token

    def tokens(self, name: str | None = None) -> list[dict[str, Any]]:
        if name:
            user = self._require(name)
            rows = self.db.query(
                "SELECT t.*, u.name AS user FROM api_tokens t JOIN users u ON u.id = t.user_id"
                " WHERE t.user_id = ? ORDER BY t.id",
                (user.id,),
            )
        else:
            rows = self.db.query(
                "SELECT t.*, u.name AS user FROM api_tokens t JOIN users u ON u.id = t.user_id"
                " ORDER BY t.id"
            )
        return [
            {k: v for k, v in dict(row).items() if k != "token_hash"} for row in rows
        ]

    def revoke_token(self, token_id: int) -> bool:
        cur = self.db.execute(
            "UPDATE api_tokens SET revoked = 1 WHERE id = ? AND revoked = 0", (token_id,)
        )
        return bool(cur.rowcount)

    def revoke_all(self, user_id: int) -> None:
        self.db.execute("UPDATE api_tokens SET revoked = 1 WHERE user_id = ?", (user_id,))
        self.db.execute("DELETE FROM web_sessions WHERE user_id = ?", (user_id,))

    def identify_token(self, token: str) -> Identity | None:
        """Who a bearer token belongs to, or None if it is not a live one."""
        if not token:
            return None
        row = self.db.query_one(
            "SELECT * FROM api_tokens WHERE token_hash = ? AND revoked = 0",
            (token_hash(token),),
        )
        if row is None:
            return None
        if row["expires_at"] and row["expires_at"] < _now().isoformat(timespec="seconds"):
            return None
        user = self.get(int(row["user_id"]))
        if user.disabled:
            return None
        self.db.execute(
            "UPDATE api_tokens SET last_used_at = datetime('now') WHERE id = ?",
            (row["id"],),
        )
        return Identity(kind="token", user=user, token_id=int(row["id"]))

    # --- browser sessions -----------------------------------------------
    def start_session(self, user: User, *, user_agent: str = "") -> str:
        token, digest = new_token()
        self.db.insert(
            "web_sessions",
            {
                "user_id": user.id,
                "token_hash": digest,
                "expires_at": (_now() + timedelta(days=self.session_days)).isoformat(
                    timespec="seconds"
                ),
                "user_agent": (user_agent or "")[:200],
            },
        )
        return token

    def identify_session(self, token: str) -> Identity | None:
        if not token:
            return None
        row = self.db.query_one(
            "SELECT * FROM web_sessions WHERE token_hash = ?", (token_hash(token),)
        )
        if row is None:
            return None
        if row["expires_at"] and row["expires_at"] < _now().isoformat(timespec="seconds"):
            self.db.execute("DELETE FROM web_sessions WHERE id = ?", (row["id"],))
            return None
        user = self.get(int(row["user_id"]))
        if user.disabled:
            return None
        self.db.execute(
            "UPDATE web_sessions SET last_seen_at = datetime('now') WHERE id = ?",
            (row["id"],),
        )
        return Identity(kind="cookie", user=user)

    def end_session(self, token: str) -> None:
        self.db.execute(
            "DELETE FROM web_sessions WHERE token_hash = ?", (token_hash(token),)
        )

    def prune(self) -> int:
        """Drop expired browser sessions. Called by the daemon's upkeep timer."""
        cur = self.db.execute(
            "DELETE FROM web_sessions WHERE expires_at IS NOT NULL AND expires_at < ?",
            (_now().isoformat(timespec="seconds"),),
        )
        return cur.rowcount or 0


class RateLimiter:
    """A fixed-window limit per caller, for the TCP listener only.

    A local assistant has no business serving a crowd, and a loop in someone's script
    should not be able to hold the GPU. Deliberately crude: one counter per address per
    minute, in memory, lost on restart. Anything cleverer would need state nobody wants
    to keep for this.
    """

    def __init__(self, per_minute: int = 120):
        self.per_minute = per_minute
        self._windows: dict[str, tuple[int, int]] = {}

    def check(self, request: Any) -> None:
        if self.per_minute <= 0:
            return
        from fastapi import HTTPException

        client = getattr(request, "client", None)
        who = getattr(client, "host", None) or "unknown"
        minute = int(_now().timestamp() // 60)
        window, count = self._windows.get(who, (minute, 0))
        if window != minute:
            window, count = minute, 0
        count += 1
        self._windows[who] = (window, count)
        if count > self.per_minute:
            raise HTTPException(
                status_code=429,
                detail=f"more than {self.per_minute} requests in a minute; slow down",
                headers={"Retry-After": "60"},
            )
