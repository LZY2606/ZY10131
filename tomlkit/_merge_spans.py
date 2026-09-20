from __future__ import annotations

import dataclasses

from typing import TYPE_CHECKING
from typing import Any


if TYPE_CHECKING:
    from tomlkit.items import Item
    from tomlkit.items import Table


@dataclasses.dataclass(frozen=True)
class SourceSpan:
    """A half-open character range ``[start, end)`` inside a source string."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid span: {self.start!r}..{self.end!r}")


@dataclasses.dataclass(frozen=True)
class BoundSpan(SourceSpan):
    """A :class:`SourceSpan` bound to the decoded source text it came from.

    Line and column numbers are 1-based; columns count characters, so a CRLF
    line ending occupies two characters of the source.
    """

    source: str = ""

    @property
    def text(self) -> str:
        return self.source[self.start : self.end]

    @property
    def start_line(self) -> int:
        return self.source.count("\n", 0, self.start) + 1

    @property
    def end_line(self) -> int:
        # The range is half-open [start, end): a trailing newline at the
        # last included character belongs to this entry but does not move the
        # end onto the next line.
        inclusive_end = self.end
        if inclusive_end > self.start and self.source[inclusive_end - 1] == "\n":
            inclusive_end -= 1
        return self.source.count("\n", 0, inclusive_end) + 1

    @property
    def start_column(self) -> int:
        return self.start - self.source.rfind("\n", 0, self.start)

    @property
    def end_column(self) -> int:
        inclusive_end = self.end
        if inclusive_end > self.start and self.source[inclusive_end - 1] == "\n":
            inclusive_end -= 1
        return inclusive_end - self.source.rfind("\n", 0, inclusive_end)

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "start_line": self.start_line,
            "start_column": self.start_column,
            "end_line": self.end_line,
            "end_column": self.end_column,
            "text": self.text,
        }


class SpanBook:
    """Collects source spans while the parser builds a document.

    The parser records one span per semantic entry: a ``key = value`` pair
    (including leading indentation and its trailing comment, when present) or
    a ``[table]`` / ``[[aot]]`` header. Composite values (inline tables,
    arrays) share the span of the entry that introduced them, while each of
    their nested entries gets its own, more precise span.
    """

    def __init__(self) -> None:
        self._spans: dict[int, SourceSpan] = {}
        # A stack because values nest: parsing the inline table of an outer
        # ``z = {...}`` entry parses inner ``p = 1`` entries in between.
        self._entry_starts: list[int] = []

    def begin_entry(self, start: int) -> None:
        self._entry_starts.append(start)

    def record(self, item: Item, start: int, end: int) -> None:
        self._spans[id(item)] = SourceSpan(start, end)

    def record_entry(self, item: Item, end: int) -> None:
        if not self._entry_starts:
            return
        start = self._entry_starts.pop()
        self._spans[id(item)] = SourceSpan(start, end)

    def record_header(self, table: Table, start: int, end: int) -> None:
        self._spans.setdefault(id(table), SourceSpan(start, end))

    def get(self, item: Item) -> SourceSpan | None:
        return self._spans.get(id(item))

    def bind(self, source: str) -> dict[int, BoundSpan]:
        return {
            item_id: BoundSpan(span.start, span.end, source)
            for item_id, span in self._spans.items()
        }
