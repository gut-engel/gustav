"""What the night learns from, and what it learned.

These are stores, not logic: the sleep steps put candidates in, the trainer and the
gate take them out. Keeping them here means every step speaks the same language about
an example, a fact, a topic and a task — and that the dataset the trainer sees is a
query, not a pile of files one step happened to write.
"""

from gustav.learn.examples import AddResult, Example, ExampleStore
from gustav.learn.facts import Fact, FactStore
from gustav.learn.skills import Skill, SkillStore

__all__ = [
    "AddResult",
    "Example",
    "ExampleStore",
    "Fact",
    "FactStore",
    "Skill",
    "SkillStore",
]
