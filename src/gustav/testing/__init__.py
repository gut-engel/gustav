"""Test support shipped with the package.

These live inside ``gustav`` rather than under ``tests/`` because child processes are
started with the ``spawn`` context and must be able to *import* their target by name.
"""
