"""Backups: nothing changes weights without a verified snapshot first (section 9)."""

from gustav.backup.guard import BackupRequired, Change, WeightChangeGuard
from gustav.backup.manifest import Manifest, hash_file
from gustav.backup.restore import RestoreResult, Restorer
from gustav.backup.retention import RetentionPolicy, plan_retention
from gustav.backup.snapshot import (
    BackupError,
    NotEnoughSpace,
    Snapshot,
    SnapshotStore,
    VerifyResult,
)

__all__ = [
    "BackupError",
    "BackupRequired",
    "Change",
    "Manifest",
    "NotEnoughSpace",
    "RestoreResult",
    "Restorer",
    "RetentionPolicy",
    "Snapshot",
    "SnapshotStore",
    "VerifyResult",
    "WeightChangeGuard",
    "hash_file",
    "plan_retention",
]
