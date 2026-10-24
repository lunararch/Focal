"""Layout planning for the split mirror workspace.

The UI deliberately keeps layout selection separate from capture state so tiles can
be reordered or rebuilt into nested splitters without restarting their mirrors.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TypeVar


class LayoutMode(StrEnum):
    ROW = "row"
    COLUMN = "column"
    TOP_PAIR = "top-pair"
    GRID = "grid"


T = TypeVar("T")


def pane_rows(items: list[T], mode: LayoutMode) -> list[list[T]]:
    """Group tiles into splitter rows for ``mode``.

    ``ROW`` and ``COLUMN`` each return one group; the caller chooses that
    splitter's orientation. ``TOP_PAIR`` gives the first two tiles a horizontal
    top splitter and places later tiles below it. ``GRID`` produces horizontal
    pairs in vertically stacked splitters.
    """
    if not items:
        return []
    if mode in (LayoutMode.ROW, LayoutMode.COLUMN):
        return [items.copy()]
    if mode is LayoutMode.TOP_PAIR:
        return [items[:2], *[[item] for item in items[2:]]]
    return [items[index : index + 2] for index in range(0, len(items), 2)]
