"""Where the CLI remembers which server it is talking to.

`ai` on your own machine talks to the daemon over its Unix socket and needs no
configuration at all. `ai` on a laptop talks to a server over HTTPS and needs two
things: the address and a token. They live in one small file.

The file holds a credential, so it is written with mode 600 and the token is never
printed back — `ai whoami` shows the server and the account, not the secret.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ENV_URL = "GUSTAV_URL"
ENV_TOKEN = "GUSTAV_TOKEN_CLIENT"


def config_path() -> Path:
    """``~/.config/gustav/client.json``, honouring XDG_CONFIG_HOME."""
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / "gustav" / "client.json"


@dataclass
class Endpoint:
    url: str | None = None
    token: str | None = None
    account: str | None = None
    #: Where the answer came from, so `ai whoami` can say.
    source: str = "none"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def remote(self) -> bool:
        return bool(self.url)

    def describe(self) -> str:
        if not self.url:
            return "the local daemon over its socket"
        who = f" as {self.account}" if self.account else ""
        return f"{self.url}{who} (from {self.source})"


def load(path: Path | None = None) -> Endpoint:
    """The configured server, if there is one. Never raises."""
    path = path or config_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Endpoint()
    if not isinstance(data, dict):
        return Endpoint()
    return Endpoint(
        url=data.get("url") or None,
        token=data.get("token") or None,
        account=data.get("account") or None,
        source=str(path),
        extra={k: v for k, v in data.items() if k not in ("url", "token", "account")},
    )


def resolve(*, url: str | None = None, token: str | None = None) -> Endpoint:
    """Which server to use, most explicit first.

    A flag beats the environment, which beats the saved file, which beats the local
    socket. The same order the rest of the project uses for configuration, so there is
    one rule to remember.
    """
    if url:
        return Endpoint(url=url, token=token, source="--url")
    env_url = os.environ.get(ENV_URL)
    if env_url:
        return Endpoint(
            url=env_url,
            token=token or os.environ.get(ENV_TOKEN),
            source=ENV_URL,
        )
    saved = load()
    if saved.url:
        if token:
            saved.token = token
        return saved
    return Endpoint(token=token)


def save(url: str, token: str, *, account: str | None = None, path: Path | None = None) -> Path:
    """Remember a server. Written 600, because it holds a token."""
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"url": url.rstrip("/"), "token": token, "account": account}
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.chmod(stat.S_IRUSR | stat.S_IWUSR)
    temporary.replace(path)
    return path


def forget(path: Path | None = None) -> bool:
    """Drop the saved server. True if there was one."""
    path = path or config_path()
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
