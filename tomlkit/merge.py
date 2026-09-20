"""Structured three-way merge for TOML documents.

Unlike a textual (line based) three-way merge, this module understands the TOML
data model: a dotted key and an explicit ``[table]`` header can resolve to the
same semantic path, an array-of-tables element is identified semantically (not
by its position), and the output is rebuilt structurally so that merging can
never produce a syntactically valid but structurally wrong document (duplicate
tables, out-of-order tables or swallowed sibling trivia).

The public entry point is :func:`merge`. Documents must be parsed with source
tracking enabled via :func:`parse_for_merge` so that conflicts can carry
``base`` / ``ours`` / ``theirs`` source evidence.
"""

from __future__ import annotations

import copy
import enum

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from tomlkit.items import AoT
from tomlkit.items import Array
from tomlkit.items import InlineTable
from tomlkit.items import Item
from tomlkit.items import Table
from tomlkit.parser import Parser


if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from tomlkit.container import Container
    from tomlkit.toml_document import TOMLDocument

    _NodeItem = Item | TOMLDocument


__all__ = [
    "Conflict",
    "ConflictKind",
    "CommentPolicy",
    "MergeResult",
    "SourceRange",
    "merge",
    "parse_for_merge",
]


Path = tuple[str, ...]
IdentityKey = str | Callable[[dict[str, Any]], Any]


class ConflictKind(str, enum.Enum):
    """The reason two changes could not be merged automatically."""

    VALUE = "value"
    """Both sides changed the same scalar/array value differently."""

    STRUCTURE = "structure"
    """The container kind changed incompatibly (table vs. array-of-tables vs.
    scalar, or an inline table vs. a regular table)."""

    COMMENT = "comment"
    """One side changed the value while the other changed only the comment
    (or both changed the comment differently), and the active policy forbids
    combining them."""

    DELETE_MODIFIED = "delete_modified"
    """One side deleted a node while the other modified it (or its value)."""

    DELETE_DESCENDANT = "delete_descendant"
    """One side deleted a container while the other modified a descendant."""

    AOT_AMBIGUOUS = "aot_ambiguous"
    """Array-of-tables elements cannot be paired without guessing (no stable
    identity key or unmatched/duplicated identity)."""

    AOT_REORDER = "aot_reorder"
    """Both sides reordered the array-of-tables differently."""


class CommentPolicy(str, enum.Enum):
    """Controls how a value change on one side and a comment change on the
    other are reconciled."""

    CONFLICT = "conflict"
    """Report a :attr:`ConflictKind.COMMENT` conflict (the safe default)."""

    MERGE = "merge"
    """Take the new value from the value-changing side and the new comment
    from the comment-changing side, when that is unambiguous."""


def parse_for_merge(source: str | bytes) -> TOMLDocument:
    """Parse a TOML document while recording source spans.

    The returned document is a normal, fully mutable ``TOMLDocument``; the only
    difference is that parsed items carry a ``source_span`` attribute used by
    :func:`merge` for conflict evidence.
    """
    return Parser(source, track_spans=True).parse()


@dataclass(frozen=True)
class SourceRange:
    """A ``[start, end)`` character range inside one input document."""

    start: int
    end: int
    text: str

    @classmethod
    def from_span(cls, source: str, span: tuple[int, int] | None) -> SourceRange | None:
        if span is None:
            return None
        start, end = span
        start = max(0, min(start, len(source)))
        end = max(start, min(end, len(source)))
        return cls(start, end, source[start:end])


@dataclass
class Conflict:
    """A single unresolved collision between the two sides."""

    path: Path
    """The semantic path (sequence of bare/quoted key strings) of the node."""

    kind: ConflictKind

    base: Item | TOMLDocument | None
    ours: Item | TOMLDocument | None
    theirs: Item | TOMLDocument | None

    base_span: SourceRange | None
    ours_span: SourceRange | None
    theirs_span: SourceRange | None

    aot_index: int | None = None
    """For an :attr:`ConflictKind.AOT_AMBIGUOUS` element conflict, the 0-based
    element index in *ours* (when applicable)."""

    candidates: tuple[Path, ...] = ()
    """Conservative pairing candidates for an unidentifiable AoT element, given
    as their semantic paths. Empty unless the merge refused to guess."""

    message: str = ""

    def __str__(self) -> str:
        loc = ".".join(self.path) or "<root>"
        return f"{self.kind.value} conflict at {loc}: {self.message}".rstrip(": ")


@dataclass
class MergeResult:
    """The outcome of a structured merge.

    ``merged`` is the auto-mergeable result (always structurally valid, and
    equal to *ours* when nothing could be applied). ``conflicts`` lists every
    unresolved collision. Call :meth:`apply` to re-resolve conflicts and build
    the final document.
    """

    merged: TOMLDocument
    conflicts: list[Conflict] = field(default_factory=list)
    _touched: list[Path] = field(default_factory=list)
    _input_docs: dict[str, TOMLDocument] = field(default_factory=dict)
    _identity: dict[Path, IdentityKey] = field(default_factory=dict)
    _comment_policy: CommentPolicy = CommentPolicy.CONFLICT
    _merger: Any = None

    @property
    def ok(self) -> bool:
        """True when there are no unresolved conflicts."""
        return not self.conflicts

    def apply(
        self,
        resolution: (
            Callable[[Conflict], object]
            | dict[Path, object]
            | None
        ) = None,
    ) -> TOMLDocument:
        """Build the final document after resolving conflicts.

        ``resolution`` is either:

        * a mapping of semantic path -> resolution value; or
        * a callable receiving each :class:`Conflict` and returning a
          resolution.

        A resolution is one of:

        * :data:`"ours"` / :data:`KEEP_OURS`,
        * :data:`"theirs"` / :data:`TAKE_THEIRS`,
        * :data:`"delete"`, or
        * an :class:`~tomlkit.items.Item` (or plain Python value, encoded via
          :func:`tomlkit.item`) used as the replacement.
        """
        from tomlkit._merge_apply import build_resolved_document

        return build_resolved_document(self, resolution)


# ---------------------------------------------------------------------------
# Logical (semantic) node model
# ---------------------------------------------------------------------------


class _Kind(str, enum.Enum):
    TABLE = "table"
    INLINE = "inline_table"
    AOT = "aot"
    ARRAY = "array"
    SCALAR = "scalar"


def _classify(item: Item | None) -> _Kind:
    if isinstance(item, Table):
        return _Kind.TABLE
    if isinstance(item, InlineTable):
        return _Kind.INLINE
    if isinstance(item, AoT):
        return _Kind.AOT
    if isinstance(item, Array):
        return _Kind.ARRAY
    return _Kind.SCALAR


@dataclass
class _Node:
    path: Path
    kind: _Kind
    item: _NodeItem
    # Logical children for TABLE / INLINE; AoT elements for AOT.
    children: dict[str, _Node] = field(default_factory=dict)
    elements: list[_Node] = field(default_factory=list)
    span: tuple[int, int] | None = None

    def is_container(self) -> bool:
        return self.kind in (_Kind.TABLE, _Kind.INLINE)


def _child_container(item: Item) -> Container:
    return item.value  # type: ignore[no-any-return]


def _span_of(item: Any) -> tuple[int, int] | None:
    if item is None:
        return None
    return getattr(item, "_source_span", None)


def _merged_span(items: Iterable[Item | None]) -> tuple[int, int] | None:
    starts: list[int] = []
    ends: list[int] = []
    for it in items:
        span = _span_of(it)
        if span is not None:
            starts.append(span[0])
            ends.append(span[1])
    if not starts:
        return None
    return (min(starts), max(ends))


def _dotted_leaf_span(table: Table) -> tuple[int, int] | None:
    """Derive the span of a dotted-key super table from its leaf KV items.

    ``a.b.c = 1`` is parsed into super tables whose header is virtual; the
    actual source range lives on the key/value line. The container stamped the
    key with the line span, so descend and collect it.
    """
    spans: list[tuple[int, int]] = []

    def walk(container: Container) -> None:
        from tomlkit.items import Key
        from tomlkit.items import Whitespace

        for key, value in container.body:
            if key is None or isinstance(value, Whitespace):
                continue
            key_span = getattr(key, "_source_span", None)
            if key_span is not None:
                spans.append(key_span)
            if isinstance(value, Table):
                walk(value.value)

    walk(table.value)
    if not spans:
        return _span_of(table)
    return (min(s[0] for s in spans), max(s[1] for s in spans))


def _table_fragment_span(table: Table) -> tuple[int, int] | None:
    span = _span_of(table)
    if span is not None:
        return span
    # Dotted-key-only super table: recover from nested KV line spans.
    return _dotted_leaf_span(table)


def _build_table_node(
    path: Path, fragments: Sequence[Table | InlineTable], kind: _Kind
) -> _Node:
    assert fragments
    node = _Node(
        path=path,
        kind=kind,
        item=fragments[0],
        span=_merged_span(fragments),
    )

    # Preserve first-occurrence order while merging out-of-order fragments.
    order: list[str] = []
    table_fragments: dict[str, list[Table]] = {}
    aot_value: dict[str, AoT] = {}
    leaf_value: dict[str, Item] = {}
    inline_kinds: dict[str, _Kind] = {}

    for fragment in fragments:
        for key, child in fragment.value.body:
            if key is None:
                continue
            name = key.key
            if name not in order:
                order.append(name)
            child_kind = _classify(child)
            inline_kinds.setdefault(name, child_kind)
            if isinstance(child, Table):
                table_fragments.setdefault(name, []).append(child)
            elif isinstance(child, AoT):
                if name in aot_value:
                    aot_value[name].body.extend(child.body)
                else:
                    aot_value[name] = child
            else:
                leaf_value[name] = child

    for name in order:
        child_path = (*path, name)
        if name in table_fragments:
            child_kind = inline_kinds[name]
            node.children[name] = _build_table_node(
                child_path, table_fragments[name], child_kind
            )
        elif name in aot_value:
            node.children[name] = _build_aot_node(child_path, aot_value[name])
        elif isinstance(leaf_value[name], InlineTable):
            inline = leaf_value[name]
            assert isinstance(inline, InlineTable)
            node.children[name] = _build_table_node(
                child_path, [inline], _Kind.INLINE
            )
        else:
            node.children[name] = _build_leaf_node(child_path, leaf_value[name])

    return node


def _build_leaf_node(path: Path, item: Item) -> _Node:
    return _Node(
        path=path,
        kind=_classify(item),
        item=item,
        span=_span_of(item),
    )


def _build_aot_node(path: Path, aot: AoT) -> _Node:
    node = _Node(
        path=path,
        kind=_Kind.AOT,
        item=aot,
        span=_span_of(aot),
    )
    for index, element in enumerate(aot.body):
        node.elements.append(
            _build_table_node((*path, str(index)), [element], _Kind.TABLE)
        )
    return node


def _build_root(doc: TOMLDocument) -> _Node:
    # The document body holds top-level fragments; same-key Table fragments are
    # out-of-order parts to merge, AoT fragments stack into one AoT.
    order: list[str] = []
    tables: dict[str, list[Table]] = {}
    leaves: dict[str, Item] = {}
    aots: dict[str, AoT] = {}

    for key, value in doc.body:
        if key is None:
            continue
        name = key.key
        if isinstance(value, Table):
            if name not in order:
                order.append(name)
            tables.setdefault(name, []).append(value)
        elif isinstance(value, AoT):
            if name not in order:
                order.append(name)
            if name in aots:
                aots[name].body.extend(value.body)
            else:
                aots[name] = value
        else:
            order.append(name)
            leaves[name] = value

    root = _Node(path=(), kind=_Kind.TABLE, item=doc, span=(0, len(doc.as_string())))
    for name in order:
        child_path = (name,)
        if name in tables:
            root.children[name] = _build_table_node(
                child_path, tables[name], _Kind.TABLE
            )
        elif name in aots:
            root.children[name] = _build_aot_node(child_path, aots[name])
        elif isinstance(leaves[name], InlineTable):
            inline_root = leaves[name]
            assert isinstance(inline_root, InlineTable)
            root.children[name] = _build_table_node(
                child_path, [inline_root], _Kind.INLINE
            )
        else:
            root.children[name] = _build_leaf_node(child_path, leaves[name])
    return root


def _node_span(node: _Node) -> tuple[int, int] | None:
    if node.span is not None:
        return node.span
    return _span_of(node.item)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _comment_of(item: Any) -> str:
    if item is None:
        return ""
    return getattr(item.trivia, "comment", "") or ""


def _value_key(item: Item | None) -> Any:
    """A structural value identity key (independent of comments/formatting)."""
    if item is None:
        return None
    try:
        return _freeze(item.unwrap())
    except Exception:  # pragma: no cover - defensive
        return item.as_string()


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return ("__dict__", tuple(sorted((k, _freeze(v)) for k, v in value.items())))
    if isinstance(value, (list, tuple)):
        return ("__list__", tuple(_freeze(v) for v in value))
    return (type(value).__name__, value)


def _same_value(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if _classify(left) is not _classify(right):
        return False
    return bool(_value_key(left) == _value_key(right))


def _is_container_kind(kind: _Kind) -> bool:
    return kind in (_Kind.TABLE, _Kind.INLINE)


def _structure_conflict(left: Item | None, right: Item | None) -> bool:
    """True when two concrete items occupy incompatible container shapes."""
    if left is None or right is None:
        return False
    kl, kr = _classify(left), _classify(right)
    if kl == kr:
        return False
    containers = {_Kind.TABLE, _Kind.INLINE, _Kind.AOT}
    if kl in containers or kr in containers:
        return True
    # One side scalar, other array/scalar: a value conflict, not structural.
    return False


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


class _Action(str, enum.Enum):
    KEEP = "keep"
    TAKE_THEIRS = "take_theirs"
    DELETE = "delete"
    MERGE_TABLE = "merge_table"
    MERGE_AOT = "merge_aot"
    COMBINE = "combine"
    CONFLICT = "conflict"


@dataclass
class _Decision:
    action: _Action
    # For MERGE_AOT: element decisions / structural operations.
    elements: list[_ElementDecision] = field(default_factory=list)
    take_item: Item | TOMLDocument | None = None
    conflict: Conflict | None = None
    comment_item: Item | TOMLDocument | None = None


@dataclass
class _ElementDecision:
    """Per-element plan for an AoT merge."""

    ours_index: int | None
    theirs_index: int | None
    base_index: int | None
    action: _Action
    # For a merged element, child keyed decisions are applied during the
    # recursive table merge.
    node: _Node | None = None
    conflict: Conflict | None = None


# ---------------------------------------------------------------------------
# Three-way merge
# ---------------------------------------------------------------------------


@dataclass
class _Sources:
    base: str
    ours: str
    theirs: str


class _Merger:
    def __init__(
        self,
        sources: _Sources,
        comment_policy: CommentPolicy,
        identity: dict[Path, IdentityKey] | None,
    ) -> None:
        self.sources = sources
        self.comment_policy = comment_policy
        self.identity = identity or {}
        self.conflicts: list[Conflict] = []
        # Flattened (path -> _Decision/operations) for the applicator, built in
        # path order.
        self.table_ops: dict[Path, dict[str, _Decision]] = {}
        # AoT element plans.
        self.aot_plans: dict[Path, list[_ElementDecision]] = {}

    # -- source evidence ----------------------------------------------------

    def _range(self, side: str, span: tuple[int, int] | None) -> SourceRange | None:
        return SourceRange.from_span(getattr(self.sources, side), span)

    def _evidence(
        self,
        base: _Node | None,
        ours: _Node | None,
        theirs: _Node | None,
    ) -> tuple[SourceRange | None, SourceRange | None, SourceRange | None]:
        return (
            self._range("base", _node_span(base) if base else None),
            self._range("ours", _node_span(ours) if ours else None),
            self._range("theirs", _node_span(theirs) if theirs else None),
        )

    def _add_conflict(
        self,
        path: Path,
        kind: ConflictKind,
        base: _Node | None,
        ours: _Node | None,
        theirs: _Node | None,
        message: str,
        **kwargs: Any,
    ) -> Conflict:
        base_item = base.item if base else None
        ours_item = ours.item if ours else None
        theirs_item = theirs.item if theirs else None
        bs, os_, ts = self._evidence(base, ours, theirs)
        conflict = Conflict(
            path=path,
            kind=kind,
            base=base_item,
            ours=ours_item,
            theirs=theirs_item,
            base_span=bs,
            ours_span=os_,
            theirs_span=ts,
            message=message,
            **kwargs,
        )
        self.conflicts.append(conflict)
        return conflict

    # -- entry point --------------------------------------------------------

    def merge(self, base: _Node, ours: _Node, theirs: _Node) -> None:
        self._merge_table(base, ours, theirs)

    def _record_table_op(self, path: Path, name: str, decision: _Decision) -> None:
        self.table_ops.setdefault(path, {})[name] = decision

    # -- tables -------------------------------------------------------------

    def _merge_table(
        self, base: _Node | None, ours: _Node | None, theirs: _Node | None
    ) -> _Decision:
        # The three nodes at this path, when present, must all be table-like.
        present = [n for n in (base, ours, theirs) if n is not None]
        if any(not n.is_container() for n in present):
            # This should only be reached from the root; scalar conflicts are
            # handled per child below.
            return _Decision(_Action.CONFLICT)

        path = next((n.path for n in (ours, theirs, base) if n is not None), ())

        names: list[str] = []
        seen: set[str] = set()
        for node in (base, ours, theirs):
            if node is None:
                continue
            for name in node.children:
                if name not in seen:
                    seen.add(name)
                    names.append(name)

        for name in names:
            b = base.children.get(name) if base else None
            o = ours.children.get(name) if ours else None
            t = theirs.children.get(name) if theirs else None
            child_path = (*path, name)
            decision = self._merge_child(child_path, b, o, t)
            self._record_table_op(path, name, decision)

        return _Decision(_Action.MERGE_TABLE)

    def _merge_child(
        self,
        path: Path,
        base: _Node | None,
        ours: _Node | None,
        theirs: _Node | None,
    ) -> _Decision:
        present = [n for n in (base, ours, theirs) if n is not None]
        kinds = {n.kind for n in present}

        # ---- pure add / delete cases -------------------------------------
        if base is None:
            if ours is None:
                # Added only by theirs (or missing everywhere).
                if theirs is None:
                    return _Decision(_Action.KEEP)
                return self._take_added(path, theirs)
            if theirs is None:
                # Added only by ours: keep ours.
                return _Decision(_Action.KEEP)
            # Added on both sides.
            return self._both_added(path, ours, theirs)

        # Base exists.
        if ours is None and theirs is None:
            # Deleted on both sides.
            return _Decision(_Action.DELETE)
        if ours is None:
            # Deleted by ours, theirs maybe changed.
            return self._deleted_side(path, base, None, theirs, deleted="ours")
        if theirs is None:
            return self._deleted_side(path, base, ours, None, deleted="theirs")

        # Present in all three.
        if ours.kind != theirs.kind:
            return self._kind_divergence(path, base, ours, theirs)
        if base.kind != ours.kind:
            # Both changed to the same shape from a different base shape.
            if ours.kind == theirs.kind and _same_value(ours.item, theirs.item):
                return self._take_if_changed_or_keep(path, base, ours, theirs)
            return self._kind_divergence(path, base, ours, theirs)

        kind = ours.kind
        if kind is _Kind.TABLE or kind is _Kind.INLINE:
            return self._merge_existing_container(path, base, ours, theirs, kind)
        if kind is _Kind.AOT:
            return self._merge_existing_aot(path, base, ours, theirs)
        return self._merge_scalar(path, base, ours, theirs)

    # -- additions / deletions ----------------------------------------------

    def _take_added(self, path: Path, node: _Node) -> _Decision:
        if node.is_container():
            # Bring over the whole added subtree; still merge internals so that
            # any further structure is wired correctly (single-sided here).
            if node.kind is _Kind.AOT:
                self._take_aot(path, node)
            else:
                self._merge_table(None, None, node)
        return _Decision(_Action.TAKE_THEIRS, take_item=node.item)

    def _take_aot(self, path: Path, node: _Node) -> None:
        plan: list[_ElementDecision] = []
        for index, element in enumerate(node.elements):
            self._merge_table(None, None, element)
            plan.append(
                _ElementDecision(
                    ours_index=None,
                    theirs_index=index,
                    base_index=None,
                    action=_Action.TAKE_THEIRS,
                    node=element,
                )
            )
        self.aot_plans[path] = plan

    def _both_added(self, path: Path, ours: _Node, theirs: _Node) -> _Decision:
        if ours.kind != theirs.kind:
            return self._conflict_structure(
                path, None, ours, theirs, "both sides added different item types"
            )
        if ours.is_container():
            if ours.kind is _Kind.AOT:
                return self._merge_both_added_aot(path, ours, theirs)
            if _same_value(ours.item, theirs.item):
                return _Decision(_Action.KEEP)
            # Merge the two added containers; conflicts inside are reported.
            return self._merge_existing_container(path, None, ours, theirs, ours.kind)
        if _same_value(ours.item, theirs.item) and _comment_of(
            ours.item
        ) == _comment_of(theirs.item):
            return _Decision(_Action.KEEP)
        if _same_value(ours.item, theirs.item):
            return self._merge_scalar(path, None, ours, theirs)
        return self._conflict_value(
            path, None, ours, theirs, "both sides added different values"
        )

    def _merge_both_added_aot(
        self, path: Path, ours: _Node, theirs: _Node
    ) -> _Decision:
        base = _Node(path=path, kind=_Kind.AOT, item=AoT([], parsed=True))
        return self._merge_existing_aot(path, base, ours, theirs, both_added=True)

    def _deleted_side(
        self,
        path: Path,
        base: _Node,
        ours: _Node | None,
        theirs: _Node | None,
        deleted: str,
    ) -> _Decision:
        survivor = ours if ours is not None else theirs
        if survivor is None:
            return _Decision(_Action.DELETE)

        modified = self._subtree_modified(base, survivor)
        if not modified:
            # Deleted on one side, unchanged on the other: deletion wins.
            return _Decision(_Action.DELETE)

        if survivor.is_container() and self._descendant_modified(base, survivor):
            kind = ConflictKind.DELETE_DESCENDANT
            msg = f"{deleted} deleted the container but the other side modified a descendant"
        else:
            kind = ConflictKind.DELETE_MODIFIED
            msg = f"{deleted} deleted the item but the other side modified it"

        conflict = self._add_conflict(
            path,
            kind,
            base,
            ours,
            theirs,
            msg,
        )
        return _Decision(_Action.CONFLICT, conflict=conflict)

    # -- modification detection ---------------------------------------------

    def _subtree_modified(self, base: _Node, other: _Node) -> bool:
        """Whether ``other`` differs from ``base`` in value, comment or shape."""
        if base.kind != other.kind:
            return True
        if not _same_value(base.item, other.item):
            return True
        if _comment_of(base.item) != _comment_of(other.item):
            return True
        if base.is_container():
            if set(base.children) != set(other.children):
                return True
            for name, child in other.children.items():
                if self._subtree_modified(base.children[name], child):
                    return True
        elif base.kind is _Kind.AOT:
            if len(base.elements) != len(other.elements):
                return True
            for b_el, o_el in zip(base.elements, other.elements):
                if self._subtree_modified(b_el, o_el):
                    return True
        return False

    def _descendant_modified(self, base: _Node, other: _Node) -> bool:
        """True when some descendant changed, even if the node itself looks
        unchanged at its own value/header level."""
        if not (base.is_container() and other.is_container()):
            return False
        for name, child in other.children.items():
            b_child = base.children.get(name)
            if b_child is None:
                return True
            if self._subtree_modified(b_child, child):
                return True
        return any(
            name not in other.children for name in base.children
        )

    # -- conflict constructors ----------------------------------------------

    def _conflict_structure(
        self,
        path: Path,
        base: _Node | None,
        ours: _Node | None,
        theirs: _Node | None,
        message: str,
    ) -> _Decision:
        conflict = self._add_conflict(
            path, ConflictKind.STRUCTURE, base, ours, theirs, message
        )
        return _Decision(_Action.CONFLICT, conflict=conflict)

    def _conflict_value(
        self,
        path: Path,
        base: _Node | None,
        ours: _Node | None,
        theirs: _Node | None,
        message: str,
        kind: ConflictKind = ConflictKind.VALUE,
    ) -> _Decision:
        conflict = self._add_conflict(
            path, kind, base, ours, theirs, message
        )
        return _Decision(_Action.CONFLICT, conflict=conflict)

    def _kind_divergence(
        self,
        path: Path,
        base: _Node | None,
        ours: _Node | None,
        theirs: _Node | None,
    ) -> _Decision:
        if ours is not None and theirs is not None:
            if _is_container_kind(ours.kind) or _is_container_kind(theirs.kind):
                # Regular table vs. inline table, table vs scalar/AoT, etc.
                return self._conflict_structure(
                    path,
                    base,
                    ours,
                    theirs,
                    "the sides changed the item to different container/value types",
                )
            return self._conflict_value(
                path,
                base,
                ours,
                theirs,
                "the sides changed the value to different types",
            )
        # One side deleted; handled by caller, but stay conservative.
        return self._conflict_structure(
            path, base, ours, theirs, "item type divergence"
        )

    # -- containers ----------------------------------------------------------

    def _take_if_changed_or_keep(
        self, path: Path, base: _Node, ours: _Node, theirs: _Node
    ) -> _Decision:
        if self._subtree_modified(base, ours):
            return _Decision(_Action.TAKE_THEIRS, take_item=ours.item)
        return _Decision(_Action.KEEP)

    def _merge_existing_container(
        self,
        path: Path,
        base: _Node | None,
        ours: _Node,
        theirs: _Node,
        kind: _Kind,
    ) -> _Decision:
        # Comment changes on a table header are tracked like scalar comments.
        if base is not None:
            header_decision = self._header_comment(path, base, ours, theirs)
            if header_decision is not None:
                self._record_table_op(path, "__header__", header_decision)

        self._merge_table(base, ours, theirs)
        return _Decision(_Action.MERGE_TABLE)

    def _header_comment(
        self, path: Path, base: _Node, ours: _Node, theirs: _Node
    ) -> _Decision | None:
        bc = _comment_of(base.item)
        oc = _comment_of(ours.item)
        tc = _comment_of(theirs.item)
        if oc == tc:
            return None
        # A header comment change on both sides: always a comment conflict.
        conflict = self._add_conflict(
            path,
            ConflictKind.COMMENT,
            base,
            ours,
            theirs,
            "both sides changed the table header comment differently",
        )
        return _Decision(_Action.CONFLICT, conflict=conflict)

    # -- scalars -------------------------------------------------------------

    def _merge_scalar(
        self,
        path: Path,
        base: _Node | None,
        ours: _Node,
        theirs: _Node,
    ) -> _Decision:
        assert base is not None
        ours_value_changed = not _same_value(base.item, ours.item)
        theirs_value_changed = not _same_value(base.item, theirs.item)
        ours_comment_changed = _comment_of(base.item) != _comment_of(ours.item)
        theirs_comment_changed = _comment_of(base.item) != _comment_of(theirs.item)

        if not ours_value_changed and not theirs_value_changed:
            # Only comments may differ.
            if ours_comment_changed and theirs_comment_changed:
                oc = _comment_of(ours.item)
                tc = _comment_of(theirs.item)
                if oc != tc:
                    return self._conflict_value(
                        path,
                        base,
                        ours,
                        theirs,
                        "both sides changed the comment differently",
                        kind=ConflictKind.COMMENT,
                    )
            return _Decision(_Action.KEEP)

        if ours_value_changed and theirs_value_changed:
            if _same_value(ours.item, theirs.item):
                # Same new value; reconcile comments below if needed.
                if ours_comment_changed and theirs_comment_changed and _comment_of(
                    ours.item
                ) != _comment_of(theirs.item):
                    return self._conflict_value(
                        path,
                        base,
                        ours,
                        theirs,
                        "same value but different comments",
                        kind=ConflictKind.COMMENT,
                    )
                return _Decision(_Action.KEEP)
            return self._conflict_value(
                path, base, ours, theirs, "both sides changed the value differently"
            )

        # Exactly one side changed the value.
        if ours_value_changed:
            value_side, comment_side = ours, theirs
            side_changed_comment = theirs_comment_changed
        else:
            value_side, comment_side = theirs, ours
            side_changed_comment = ours_comment_changed

        if side_changed_comment:
            if self.comment_policy is CommentPolicy.CONFLICT:
                return self._conflict_value(
                    path,
                    base,
                    ours,
                    theirs,
                    "one side changed the value and the other changed the comment",
                    kind=ConflictKind.COMMENT,
                )
            # MERGE policy: value from the value-changing side, comment from
            # the comment-changing side (applied during document build).
            combined = _Decision(
                _Action.COMBINE,
                take_item=value_side.item,
            )
            combined.comment_item = comment_side.item
            return combined

        return _Decision(_Action.TAKE_THEIRS, take_item=value_side.item)

    # -- array of tables -----------------------------------------------------

    def _identity_for(self, path: Path) -> IdentityKey | None:
        # Exact path match first, then a parent-AoT path match so callers can
        # register ``("servers",)`` once.
        if path in self.identity:
            return self.identity[path]
        for size in range(len(path) - 1, 0, -1):
            candidate = path[:size]
            if candidate in self.identity:
                return self.identity[candidate]
        return None

    @staticmethod
    def _identity_value(element: _Node, key: IdentityKey) -> Any:
        unwrapped = element.item.unwrap()
        if callable(key):
            return key(unwrapped)
        if isinstance(unwrapped, dict):
            sentinel = object()
            value = unwrapped.get(key, sentinel)
            if value is sentinel:
                return _MISSING
            return _freeze(value)
        return _MISSING

    def _index_aot(
        self, node: _Node, id_key: IdentityKey | None
    ) -> dict[Any, list[int]]:
        index: dict[Any, list[int]] = {}
        for i, element in enumerate(node.elements):
            ident: Any = i
            if id_key is not None:
                ident = self._identity_value(element, id_key)
            index.setdefault(ident, []).append(i)
        return index

    def _merge_existing_aot(
        self,
        path: Path,
        base: _Node,
        ours: _Node,
        theirs: _Node,
        both_added: bool = False,
    ) -> _Decision:
        id_key = self._identity_for(path)

        # Without an identity key, only the conservative cases are auto
        # merged; any element-level change on both sides is reported as a
        # candidate conflict instead of a positional guess.
        if id_key is None:
            return self._merge_aot_positional(path, base, ours, theirs, both_added)

        return self._merge_aot_identified(
            path, base, ours, theirs, id_key, both_added
        )

    # -- positional (no stable identity) ------------------------------------

    def _merge_aot_positional(
        self,
        path: Path,
        base: _Node,
        ours: _Node,
        theirs: _Node,
        both_added: bool,
    ) -> _Decision:
        def changed(node: _Node) -> bool:
            return self._subtree_modified(base, node)

        ours_changed = changed(ours)
        theirs_changed = changed(theirs)

        # Only one side (or neither) touched the array: safe to take as-is,
        # and no positional pairing is required.
        if not ours_changed and not theirs_changed:
            self.aot_plans[path] = []
            return _Decision(_Action.KEEP)
        if ours_changed and not theirs_changed:
            self.aot_plans[path] = []
            return _Decision(_Action.KEEP)
        if theirs_changed and not ours_changed:
            self.aot_plans[path] = []
            return _Decision(_Action.TAKE_THEIRS, take_item=theirs.item)

        # Both sides touched an identity-less AoT. Pairing by position would
        # be a guess; report the whole array conservatively with candidate
        # evidence instead.
        candidate_paths = (ours.path, theirs.path)
        conflict = self._add_conflict(
            path,
            ConflictKind.AOT_AMBIGUOUS,
            base,
            ours,
            theirs,
            "both sides modified an array of tables without an identity key; "
            "refusing to match elements by position",
            candidates=candidate_paths,
        )
        self.aot_plans[path] = []
        return _Decision(_Action.MERGE_AOT, conflict=conflict)

    def _aot_element(
        self,
        parent_path: Path,
        index: int,
        base: _Node | None,
        ours: _Node | None,
        theirs: _Node | None,
        identifiable: bool,
    ) -> _ElementDecision:
        elem_path: Path = (*parent_path, str(index))
        # Mirror the table child decision logic but record an element conflict
        # carrying conservative candidate paths.
        if base is None:
            if ours is None and theirs is not None:
                self._merge_table(None, None, theirs)
                return _ElementDecision(
                    None, index, None, _Action.TAKE_THEIRS, node=theirs
                )
            if theirs is None and ours is not None:
                return _ElementDecision(
                    index, None, None, _Action.KEEP, node=ours
                )
            if ours is not None and theirs is not None:
                if _same_value(ours.item, theirs.item):
                    return _ElementDecision(
                        index, index, None, _Action.KEEP, node=ours
                    )
                conflict = self._ambiguous_element(
                    elem_path, None, ours, theirs
                )
                return _ElementDecision(
                    index, index, None, _Action.CONFLICT, ours, conflict
                )
            return _ElementDecision(None, None, None, _Action.KEEP)

        # Base element exists.
        if ours is None and theirs is None:
            return _ElementDecision(index, index, index, _Action.DELETE)
        if ours is None or theirs is None:
            survivor = ours if ours is not None else theirs
            assert survivor is not None
            if self._subtree_modified(base, survivor):
                conflict = self._ambiguous_element(
                    elem_path, base, ours, theirs
                )
                return _ElementDecision(
                    index,
                    index if ours is not None else None,
                    index,
                    _Action.CONFLICT,
                    ours,
                    conflict,
                )
            if ours is None:
                return _ElementDecision(None, index, index, _Action.DELETE)
            return _ElementDecision(index, None, index, _Action.KEEP, ours)

        # Element present in all three.
        o_changed = self._subtree_modified(base, ours)
        t_changed = self._subtree_modified(base, theirs)
        if not o_changed and not t_changed:
            return _ElementDecision(index, index, index, _Action.KEEP, ours)
        if o_changed and t_changed:
            conflict = self._ambiguous_element(elem_path, base, ours, theirs)
            return _ElementDecision(
                index, index, index, _Action.CONFLICT, ours, conflict
            )
        if o_changed:
            return _ElementDecision(index, None, index, _Action.KEEP, ours)
        # theirs-only change
        self._merge_table(base, ours, theirs)
        return _ElementDecision(
            index, index, index, _Action.TAKE_THEIRS, node=theirs
        )

    def _ambiguous_element(
        self,
        elem_path: Path,
        base: _Node | None,
        ours: _Node | None,
        theirs: _Node | None,
    ) -> Conflict:
        candidates: list[Path] = []
        if theirs is not None:
            candidates.append(theirs.path)
        if ours is not None and ours.path not in candidates:
            candidates.append(ours.path)
        return self._add_conflict(
            elem_path,
            ConflictKind.AOT_AMBIGUOUS,
            base,
            ours,
            theirs,
            "array-of-tables elements cannot be aligned without a stable identity key",
            aot_index=elem_path[-1] if elem_path else None,
            candidates=tuple(candidates),
        )

    # -- identified AoT ------------------------------------------------------

    def _merge_aot_identified(
        self,
        path: Path,
        base: _Node,
        ours: _Node,
        theirs: _Node,
        id_key: IdentityKey,
        both_added: bool,
    ) -> _Decision:
        base_idx = self._index_aot(base, id_key)
        ours_idx = self._index_aot(ours, id_key)
        theirs_idx = self._index_aot(theirs, id_key)

        # Duplicate identities make pairing ambiguous on that side.
        duplicate_ids = {
            ident
            for ident, positions in (*ours_idx.items(), *theirs_idx.items())
            if len(positions) > 1
        }
        if duplicate_ids and not both_added:
            conflict = self._add_conflict(
                path,
                ConflictKind.AOT_AMBIGUOUS,
                base,
                ours,
                theirs,
                f"duplicate identity values {sorted(map(str, duplicate_ids))}",
                candidates=(ours.path, theirs.path),
            )
            self.aot_plans[path] = []
            return _Decision(_Action.MERGE_AOT, conflict=conflict)

        def unique(index: dict[Any, list[int]]) -> dict[Any, int]:
            return {ident: pos[0] for ident, pos in index.items()}

        b_u, o_u, t_u = unique(base_idx), unique(ours_idx), unique(theirs_idx)

        plan: list[_ElementDecision] = []

        # Choose the output element order. When exactly one side reordered the
        # shared elements relative to base, take that side's order; otherwise
        # (both unchanged, or a two-way divergent reorder handled below) use
        # ours. Elements added only by the non-ordering side are appended.
        # Reorder analysis only concerns identities shared by all three; a
        # deleted identity does not define an order.
        shared_all = [ident for ident in b_u if ident in o_u and ident in t_u]
        base_shared = [ident for ident in b_u if ident in shared_all]
        ours_shared = sorted(shared_all, key=lambda k: o_u[k])
        theirs_shared = sorted(shared_all, key=lambda k: t_u[k])
        ours_reordered = ours_shared != base_shared
        theirs_reordered = theirs_shared != base_shared
        if theirs_reordered and not ours_reordered:
            leading_side = "theirs"
        else:
            leading_side = "ours"

        # Output ordering: walk the leading side in its position order (this
        # covers its shared and added identities), then append anything the
        # other side has that has not appeared yet (theirs-only adds, or
        # identities the leading side deleted). Every identity is still paired
        # by identity, never by position.
        def position_order(
            node: _Node, node_index: dict[Any, list[int]]
        ) -> list[Any]:
            return sorted(node_index, key=lambda k: node_index[k][0])

        if leading_side == "theirs":
            first_order = position_order(theirs, theirs_idx)
            second_order = position_order(ours, ours_idx)
        else:
            first_order = position_order(ours, ours_idx)
            second_order = position_order(theirs, theirs_idx)
        ordered_identities = list(first_order)
        for ident in second_order:
            if ident not in ordered_identities:
                ordered_identities.append(ident)

        emitted: set[Any] = set()

        def emit(ident: Any) -> None:
            if ident in emitted:
                return
            emitted.add(ident)
            o_pos = o_u.get(ident)
            t_pos = t_u.get(ident)
            b_pos = b_u.get(ident)
            b_el = base.elements[b_pos] if b_pos is not None else None
            o_el = ours.elements[o_pos] if o_pos is not None else None
            t_el = theirs.elements[t_pos] if t_pos is not None else None
            plan.append(
                self._identified_element(
                    path, ident, b_el, o_el, t_el, o_pos, t_pos, b_pos
                )
            )

        for ident in ordered_identities:
            emit(ident)

        def append_side(node: _Node, index_map: dict[Any, int], side: str) -> None:
            for ident, pos in index_map.items():
                if ident in emitted or ident in b_u:
                    continue
                emitted.add(ident)
                element = node.elements[pos]
                if side == "theirs":
                    self._merge_table(None, None, element)
                    plan.append(
                        _ElementDecision(
                            None, pos, None, _Action.TAKE_THEIRS, node=element
                        )
                    )
                else:
                    plan.append(
                        _ElementDecision(pos, None, None, _Action.KEEP, element)
                    )

        # Emit side-specific additions following the ordering side, then the
        # other side.
        if leading_side == "theirs":
            append_side(theirs, t_u, "theirs")
            append_side(ours, o_u, "ours")
        else:
            append_side(ours, o_u, "ours")
            append_side(theirs, t_u, "theirs")

        # Elements only in base (deleted on both sides).
        for ident, b_pos in b_u.items():
            if ident in o_u or ident in t_u or ident in emitted:
                continue
            b_el = base.elements[b_pos]
            plan.append(
                _ElementDecision(b_pos, None, b_pos, _Action.DELETE, node=b_el)
            )

        self.aot_plans[path] = plan

        # Detect both-side reordering relative to base, restricted to the
        # elements shared by all three (adds/deletes do not define an order
        # change). When both sides change the shared relative order and end up
        # disagreeing, alignment is ambiguous.
        shared = [ident for ident in b_u if ident in o_u and ident in t_u]
        base_shared = [ident for ident in b_u if ident in shared]
        ours_shared = sorted(shared, key=lambda k: o_u[k])
        theirs_shared = sorted(shared, key=lambda k: t_u[k])
        ours_moved = ours_shared != base_shared
        theirs_moved = theirs_shared != base_shared
        if ours_moved and theirs_moved and ours_shared != theirs_shared:
            conflict = self._add_conflict(
                path,
                ConflictKind.AOT_REORDER,
                base,
                ours,
                theirs,
                "both sides reordered the array of tables differently",
                candidates=(ours.path, theirs.path),
            )
            return _Decision(_Action.MERGE_AOT, conflict=conflict)

        return _Decision(_Action.MERGE_AOT)

    def _identified_element(
        self,
        parent_path: Path,
        ident: Any,
        base: _Node | None,
        ours: _Node | None,
        theirs: _Node | None,
        o_pos: int | None,
        t_pos: int | None,
        b_pos: int | None,
    ) -> _ElementDecision:
        ident_part = ident if isinstance(ident, str) else str(ident)
        elem_path: Path = (*parent_path, ident_part)

        if base is None:
            # New element: present on one or both sides.
            if ours is not None and theirs is None:
                return _ElementDecision(o_pos, None, None, _Action.KEEP, ours)
            if theirs is not None and ours is None:
                self._merge_table(None, None, theirs)
                return _ElementDecision(
                    None, t_pos, None, _Action.TAKE_THEIRS, theirs
                )
            if ours is None and theirs is None:
                return _ElementDecision(None, None, None, _Action.KEEP)
            if _same_value(ours.item, theirs.item):  # type: ignore[union-attr]
                return _ElementDecision(o_pos, t_pos, None, _Action.KEEP, ours)
            conflict = self._add_conflict(
                elem_path,
                ConflictKind.VALUE,
                None,
                ours,
                theirs,
                "both sides added an element with the same identity but different content",
            )
            return _ElementDecision(
                o_pos, t_pos, None, _Action.CONFLICT, ours, conflict
            )

        # Base element exists.
        if ours is None and theirs is None:
            return _ElementDecision(o_pos, t_pos, b_pos, _Action.DELETE, base)
        if ours is None or theirs is None:
            survivor = ours if ours is not None else theirs
            assert survivor is not None
            if self._subtree_modified(base, survivor):
                kind = (
                    ConflictKind.DELETE_DESCENDANT
                    if self._descendant_modified(base, survivor)
                    else ConflictKind.DELETE_MODIFIED
                )
                conflict = self._add_conflict(
                    elem_path,
                    kind,
                    base,
                    ours,
                    theirs,
                    "one side deleted the element while the other modified it",
                )
                return _ElementDecision(
                    o_pos, t_pos, b_pos, _Action.CONFLICT, ours, conflict
                )
            if ours is None:
                return _ElementDecision(None, t_pos, b_pos, _Action.DELETE, base)
            return _ElementDecision(o_pos, None, b_pos, _Action.KEEP, ours)

        # Present in all three: merge element internals.
        o_changed = self._subtree_modified(base, ours)
        t_changed = self._subtree_modified(base, theirs)
        if not o_changed:
            if not t_changed:
                return _ElementDecision(o_pos, t_pos, b_pos, _Action.KEEP, ours)
            self._merge_table(base, ours, theirs)
            return _ElementDecision(
                o_pos, t_pos, b_pos, _Action.TAKE_THEIRS, theirs
            )
        if not t_changed:
            return _ElementDecision(o_pos, t_pos, b_pos, _Action.KEEP, ours)

        # Both changed: recurse; per-field conflicts are recorded there.
        self._merge_table(base, ours, theirs)
        return _ElementDecision(o_pos, t_pos, b_pos, _Action.MERGE_TABLE, ours)


_MISSING = object()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def merge(
    base: str | TOMLDocument,
    ours: str | TOMLDocument,
    theirs: str | TOMLDocument,
    *,
    comment_policy: CommentPolicy | str = CommentPolicy.CONFLICT,
    identity: dict[str | Path | Sequence[str], IdentityKey] | None = None,
) -> MergeResult:
    """Structurally three-way merge two TOML revisions of a common ``base``.

    :param base: the common ancestor, as source text or a document parsed with
        :func:`parse_for_merge`.
    :param ours: the local revision. Untouched regions keep *ours*' exact
        representation.
    :param theirs: the incoming revision.
    :param comment_policy: controls whether a value change on one side combined
        with a comment-only change on the other merges automatically or raises
        a :attr:`ConflictKind.COMMENT` conflict.
    :param identity: maps an array-of-tables path (tuple of keys, or a dotted
        string) to an identity key name (or callable receiving the unwrapped
        element dict and returning a stable identity). AoT paths without an
        entry are never positionally guessed: conflicting modifications are
        reported as :attr:`ConflictKind.AOT_AMBIGUOUS` with candidates.

    The returned :class:`MergeResult` always holds a structurally valid
    document plus every conflict with ``base`` / ``ours`` / ``theirs`` source
    evidence. Call :meth:`MergeResult.apply` with resolutions to produce the
    final document.
    """
    from tomlkit._merge_apply import build_initial_document

    sources, docs = _coerce_inputs(base, ours, theirs)

    normalized_policy = (
        CommentPolicy(comment_policy) if not isinstance(comment_policy, CommentPolicy)
        else comment_policy
    )
    normalized_identity = _normalize_identity(identity)

    base_root = _build_root(docs["base"])
    ours_root = _build_root(docs["ours"])
    theirs_root = _build_root(docs["theirs"])

    merger = _Merger(sources, normalized_policy, normalized_identity)
    merger.merge(base_root, ours_root, theirs_root)

    merged, touched = build_initial_document(
        docs["ours"],
        docs["theirs"],
        merger,
    )

    return MergeResult(
        merged=merged,
        conflicts=list(merger.conflicts),
        _touched=touched,
        _input_docs=docs,
        _identity=normalized_identity,
        _comment_policy=normalized_policy,
        _merger=merger,
    )


def _coerce_inputs(
    base: str | TOMLDocument,
    ours: str | TOMLDocument,
    theirs: str | TOMLDocument,
) -> tuple[_Sources, dict[str, TOMLDocument]]:
    from tomlkit.toml_document import TOMLDocument as _Doc

    raw: dict[str, str] = {}
    docs: dict[str, TOMLDocument] = {}
    for name, value in (("base", base), ("ours", ours), ("theirs", theirs)):
        if isinstance(value, _Doc):
            raw[name] = value.as_string()
            docs[name] = value
        elif isinstance(value, str):
            raw[name] = value
            docs[name] = parse_for_merge(value)
        else:
            msg = f"{name} must be TOML source text or a TOMLDocument"
            raise TypeError(msg)

    sources = _Sources(raw["base"], raw["ours"], raw["theirs"])
    return sources, docs


def _normalize_identity(
    identity: dict[str | Path | Sequence[str], IdentityKey] | None,
) -> dict[Path, IdentityKey]:
    if not identity:
        return {}
    normalized: dict[Path, IdentityKey] = {}
    for key, value in identity.items():
        if isinstance(key, str):
            path: Path = tuple(part for part in key.split(".") if part)
        else:
            path = tuple(key)
        normalized[path] = value
    return normalized
