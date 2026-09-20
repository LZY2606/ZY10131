"""Structured three-way merge for TOML documents.

Unlike a textual three-way merge, this module merges the semantic TOML
trees and only then renders the result, so unrelated regions keep their
original representation.

The entry point is :func:`merge3`.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import math

from collections.abc import Callable
from collections.abc import Hashable
from collections.abc import Mapping
from typing import Any
from typing import cast

from tomlkit._merge_spans import BoundSpan
from tomlkit._merge_spans import SpanBook
from tomlkit.container import Container
from tomlkit.container import OutOfOrderTableProxy
from tomlkit.items import AoT
from tomlkit.items import InlineTable
from tomlkit.items import Item
from tomlkit.items import Table
from tomlkit.items import Whitespace
from tomlkit.parser import Parser
from tomlkit.toml_document import TOMLDocument


class ContainerType(enum.Enum):
    """The structural kind a semantic node can have."""

    TABLE = "table"
    INLINE_TABLE = "inline-table"
    AOT = "array-of-tables"
    VALUE = "value"


class ConflictKind(enum.Enum):
    """Why a path could not be merged automatically."""

    #: Both sides changed the same scalar/array value incompatibly.
    VALUE = "value"
    #: Container types at one path changed incompatibly (table vs. inline
    #: table vs. array of tables vs. scalar), or a table/AoT subtree was
    #: retargeted where the other side edited descendants.
    STRUCTURE = "structure"
    #: The comment attached to an entry was edited on both sides (or a value
    #: edit and a comment-only edit collided under the CONFLICT policy).
    COMMENT = "comment"
    #: A value/container was deleted by one side and replaced by the other.
    DELETE = "delete"
    #: A container was deleted by one side while descendants were modified
    #: by the other.
    DELETE_DESCENDANTS = "delete-descendants"
    #: An array of tables changed without a usable identity policy.
    AOT_CANDIDATES = "aot-candidates"


class Resolution(enum.Enum):
    """Built-in resolutions for a :class:`Conflict`."""

    OURS = "ours"
    THEIRS = "theirs"


class CommentConflictPolicy(enum.Enum):
    """Controls value-edit vs. comment-edit collisions on one entry.

    When one side changes an entry value and the other side changes only its
    comment:

    * ``MERGE`` (default): keep the changed value and the changed comment.
    * ``CONFLICT``: report a :attr:`ConflictKind.COMMENT` conflict instead.
    """

    MERGE = "merge"
    CONFLICT = "conflict"


#: Maps ``(aot-path, element-table)`` to a stable hashable identity, or
#: ``None`` when the element cannot be identified reliably.
IdentityPolicy = Callable[[tuple[str, ...], Table], Hashable | None]

#: Any concrete semantic node (a document is the root map node).
Node = Item | TOMLDocument


def identity_key(*key: str) -> IdentityPolicy:
    """Build an :data:`IdentityPolicy` from one key or a dotted key path.

    ``identity_key("name")`` identifies ``[[servers]]`` elements by their
    ``name`` field; ``identity_key("meta", "id")`` (or
    ``identity_key("meta.id")``) uses the nested ``meta.id`` field. Elements
    missing the key get no identity, so the array of tables is reported as
    candidates rather than guessed at.
    """
    dotted: tuple[str, ...]
    if len(key) == 1 and "." in key[0]:
        dotted = tuple(key[0].split("."))
    else:
        dotted = tuple(key)

    def _policy(_aot_path: tuple[str, ...], element: Table) -> Hashable | None:
        cur: Any = element
        for part in dotted:
            if part not in cur:
                return None
            cur = cur[part]
        if isinstance(cur, Item):
            cur = cur.unwrap()
        if isinstance(cur, Hashable):
            return cur
        return None

    return _policy


@dataclasses.dataclass(frozen=True)
class Conflict:
    """A path that could not be merged automatically.

    The three ``*_value`` fields are the raw tomlkit items at :attr:`path`,
    or ``None`` when the path did not exist on that side (``None`` at base
    means "added"; ``None`` at ours/theirs means "deleted"). The three spans
    point at the exact source evidence for each side; they are ``None`` only
    for paths that have no literal source, such as an implicit super-table
    created by a dotted key.
    """

    path: tuple[str, ...]
    kind: ConflictKind
    base_value: Node | None
    ours_value: Node | None
    theirs_value: Node | None
    base_span: BoundSpan | None
    ours_span: BoundSpan | None
    theirs_span: BoundSpan | None
    message: str = ""

    @property
    def path_string(self) -> str:
        """Dotted rendering of :attr:`path`."""
        return ".".join(self.path)

    def __str__(self) -> str:
        loc = self.path_string or "<root>"
        tail = f": {self.message}" if self.message else ""
        return f"{self.kind.value} conflict at {loc}{tail}"


@dataclasses.dataclass
class _Side:
    name: str
    doc: TOMLDocument
    source: str
    spans: dict[int, BoundSpan]

    def span_of(self, item: Node | None) -> BoundSpan | None:
        if item is None:
            return None
        direct = self.spans.get(id(item))
        if direct is not None:
            return direct
        # An AoT renders as a sequence of element tables; use the span from
        # its first header through its last element as evidence.
        if isinstance(item, AoT) and item.body:
            first = self.spans.get(id(item.body[0]))
            last = self.spans.get(id(item.body[-1]))
            if first is not None and last is not None:
                return BoundSpan(first.start, last.end, first.source)
            return first
        # A merged/proxy table: evidence from its first concrete part.
        internal = getattr(item, "_internal_container", None)
        if internal is not None:
            for _, body_item in internal.body:
                span = self.spans.get(id(body_item))
                if span is not None:
                    return span
        return None


def parse_with_spans(
    source: str | bytes,
) -> tuple[TOMLDocument, dict[int, BoundSpan]]:
    """Parse TOML and return the document together with its source spans.

    The returned mapping is keyed by ``id(item)`` for each parsed item; it is
    only meaningful for the lifetime of the returned document.
    """
    text = source if isinstance(source, str) else source.decode("utf-8")
    book = SpanBook()
    doc = Parser(text).with_span_book(book).parse()
    return doc, book.bind(text)


@dataclasses.dataclass
class MergeResult:
    """The outcome of :func:`merge3`.

    :attr:`document` always renders valid TOML. While conflicts remain it is
    a *preview*: unresolved entries fall back to the ours side. Call
    :meth:`resolve` with per-path resolutions (or custom tomlkit items) and
    re-inspect :attr:`conflicts`; :meth:`merged_document` only returns a
    document once every conflict has been resolved.
    """

    _sources: tuple[str, str, str]
    _policy: CommentConflictPolicy
    _identity: IdentityPolicy | None
    conflicts: list[Conflict]
    document: TOMLDocument
    _resolutions: dict[tuple[str, ...], Item | Resolution] = dataclasses.field(
        default_factory=dict
    )

    @property
    def clean(self) -> bool:
        """True when no conflicts are left."""
        return not self.conflicts

    def merged_document(self) -> TOMLDocument:
        """Return the merged document, raising if conflicts remain."""
        if self.conflicts:
            raise TOMLKitMergeError(
                f"{len(self.conflicts)} unresolved conflict(s): "
                + ", ".join(str(c) for c in self.conflicts)
            )
        return self.document

    def resolve(
        self,
        resolutions: Mapping[
            tuple[str, ...] | str, Resolution | Item | object
        ]
        | None = None,
        /,
        **kwargs: Resolution | Item | object,
    ) -> MergeResult:
        """Apply resolutions and recompute the merged document.

        Paths may be given as tuples or as dotted strings
        (``"servers.@alpha.port"``; ``@<id>`` or an integer addresses an
        array-of-tables element). Values may be a :class:`Resolution` or a
        custom tomlkit :class:`~tomlkit.items.Item` (use
        :func:`tomlkit.item` / :func:`tomlkit.integer` / ... to build one).
        """
        merged: dict[tuple[str, ...], Item | Resolution] = dict(self._resolutions)
        if resolutions:
            for path, value in resolutions.items():
                merged[_normalize_path(path)] = _as_resolution(value)
        for path, value in kwargs.items():
            merged[_normalize_path(path)] = _as_resolution(value)
        return _compute(
            self._sources,
            policy=self._policy,
            identity=self._identity,
            resolutions=merged,
        )


class TOMLKitMergeError(RuntimeError):
    """Raised when a clean document is requested with conflicts present."""


def _normalize_path(path: tuple[str, ...] | str) -> tuple[str, ...]:
    if isinstance(path, str):
        return tuple(p for p in path.split(".") if p != "")
    return tuple(path)


def _as_resolution(value: Resolution | Item | object) -> Item | Resolution:
    if isinstance(value, (Resolution, Item)):
        return value
    from tomlkit.items import item as _item

    return _item(value)


# ---------------------------------------------------------------------------
# Semantic view
# ---------------------------------------------------------------------------

def _node_type(item: Node) -> ContainerType:
    if isinstance(item, (TOMLDocument, OutOfOrderTableProxy)):
        return ContainerType.TABLE
    if isinstance(item, AoT):
        return ContainerType.AOT
    if isinstance(item, InlineTable):
        return ContainerType.INLINE_TABLE
    if isinstance(item, Table):
        return ContainerType.TABLE
    return ContainerType.VALUE


def _is_map_node(item: object) -> bool:
    return isinstance(item, (Table, InlineTable, Container))


def _container_of(item: Any) -> Container:
    """The concrete backing container of a map-like semantic node."""
    if isinstance(item, Container):
        return item
    internal = getattr(item, "_internal_container", None)
    if internal is not None:
        return internal  # type: ignore[no-any-return]
    return item.value  # type: ignore[no-any-return]


def _map_keys(item: Any) -> list[str]:
    seen: dict[str, None] = {}
    container = _container_of(item) if _is_map_node(item) else None
    if container is None:
        return []
    for body_key, _ in container.body:
        if body_key is not None:
            seen[body_key.key] = None
    return list(seen)


def _comment_of(item: Item | Any) -> str:
    trivia = getattr(item, "trivia", None)
    if trivia is None:
        return ""
    return trivia.comment or ""


def _set_comment(item: Item | Any, comment: str) -> None:
    item.trivia.comment = comment


def _transfer_comment(destination: Item | Any, source: Item | Any) -> None:
    """Carry an entry comment onto another item, keeping its separator."""
    comment = _comment_of(source)
    if comment:
        destination.trivia.comment = comment
        if not destination.trivia.comment_ws:
            source_ws = getattr(source.trivia, "comment_ws", "") or " "
            destination.trivia.comment_ws = source_ws


def _unwrap(value: Any) -> Any:
    if hasattr(value, "unwrap"):
        try:
            return value.unwrap()
        except (AttributeError, TypeError, ValueError):  # defensive
            return value
    return value


def _equal_unwrapped(left: Any, right: Any) -> bool:
    if (
        isinstance(left, float)
        and isinstance(right, float)
        and math.isnan(left)
        and math.isnan(right)
    ):
        return True
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False
        return all(_equal_unwrapped(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(_equal_unwrapped(a, b) for a, b in zip(left, right))
    return bool(left == right)


def _values_equal(left: Any, right: Any) -> bool:
    """NaN-aware semantic equality of unwrapped Python values."""
    return bool(_equal_unwrapped(_unwrap(left), _unwrap(right)))


def _entry_equal(
    ours: Item | Any,
    theirs: Item | Any,
    *,
    compare_comments: bool = True,
) -> bool:
    """Semantic equality of two map entries / values."""
    if not _values_equal(ours, theirs):
        return False
    return not (
        compare_comments and _comment_of(ours) != _comment_of(theirs)
    )


def _container_touched(base: Node | None, changed: Node | None) -> bool:
    """Whether a map/AoT node's contents or comments differ from base."""
    if base is None or changed is None:
        return True
    if _node_type(base) != _node_type(changed):
        return True
    return not _entry_equal(base, changed)


# ---------------------------------------------------------------------------
# Merge analysis
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Context:
    sides: dict[str, _Side]
    policy: CommentConflictPolicy
    identity: IdentityPolicy | None
    resolutions: dict[tuple[str, ...], Item | Resolution]
    conflicts: list[Conflict]

    def span_of(self, side: str, item: Node | None) -> BoundSpan | None:
        return self.sides[side].span_of(item)

    def report(
        self,
        path: tuple[str, ...],
        kind: ConflictKind,
        base: Node | None,
        ours: Node | None,
        theirs: Node | None,
        message: str = "",
    ) -> None:
        self.conflicts.append(
            Conflict(
                path=path,
                kind=kind,
                base_value=base,
                ours_value=ours,
                theirs_value=theirs,
                base_span=self.span_of("base", base),
                ours_span=self.span_of("ours", ours),
                theirs_span=self.span_of("theirs", theirs),
                message=message,
            )
        )

    def resolution_for(self, path: tuple[str, ...]) -> Item | Resolution | None:
        return self.resolutions.get(path)


def _merged_copy(item: Node) -> Node:
    """Deep-copy an item taken from an input document for merged output."""
    return copy.deepcopy(item)


_CONTAINER_KINDS = (
    ContainerType.TABLE,
    ContainerType.INLINE_TABLE,
    ContainerType.AOT,
)


def _resolved_node(
    ctx: _Context,
    path: tuple[str, ...],
    base: Node | None,
    ours: Node | None,
    theirs: Node | None,
) -> Node | None:
    """Apply an explicit resolution for ``path`` (``None`` => delete)."""
    resolution = ctx.resolution_for(path)
    if resolution is None:
        # Unresolved conflict: conservative preview keeps the ours state.
        return _merged_copy(ours) if ours is not None else None
    if resolution is Resolution.OURS:
        chosen = ours
    elif resolution is Resolution.THEIRS:
        chosen = theirs
    elif isinstance(resolution, Item):
        return _merged_copy(resolution)
    else:  # pragma: no cover - exhaustive
        raise TypeError(resolution)
    return _merged_copy(chosen) if chosen is not None else None


def _merge_node(
    ctx: _Context,
    path: tuple[str, ...],
    base: Node | None,
    ours: Node | None,
    theirs: Node | None,
) -> Node | None:
    """Three-way-merge the node at ``path``.

    Returns the item to render, or ``None`` to delete the path. Conflicts are
    appended to ``ctx.conflicts``.
    """
    if ctx.resolution_for(path) is not None:
        return _resolved_node(ctx, path, base, ours, theirs)

    b_kind = _node_type(base) if base is not None else None
    o_kind = _node_type(ours) if ours is not None else None
    t_kind = _node_type(theirs) if theirs is not None else None

    # Whole-node deletion on one side.
    if ours is None or theirs is None:
        return _merge_deletion(ctx, path, base, ours, theirs, b_kind, o_kind, t_kind)

    # Both sides exist.
    ours_changed_kind = b_kind is None or o_kind != b_kind
    theirs_changed_kind = b_kind is None or t_kind != b_kind
    if o_kind != t_kind:
        promoted = _promote_for_compatible_merge(
            base, b_kind, ours, o_kind, theirs, t_kind
        )
        if promoted is not None:
            ours, theirs, common_kind = promoted
            o_kind = t_kind = common_kind
        elif ours_changed_kind or theirs_changed_kind:
            ctx.report(
                path,
                ConflictKind.STRUCTURE,
                base,
                ours,
                theirs,
                "incompatible structural types at the same path",
            )
            return _resolved_node(ctx, path, base, ours, theirs)

    # Kinds agree here (possibly after promotion).
    assert o_kind is not None
    if o_kind in (ContainerType.TABLE, ContainerType.INLINE_TABLE):
        return _merge_map_contents(ctx, path, base, ours, theirs)
    if o_kind is ContainerType.AOT:
        return _merge_aot(ctx, path, base, ours, theirs)

    # Both are plain values (scalars or plain arrays).
    return _merge_value(ctx, path, base, ours, theirs)


_PROMOTIONS: dict[frozenset[ContainerType], ContainerType] = {
    frozenset(
        {ContainerType.INLINE_TABLE, ContainerType.TABLE}
    ): ContainerType.TABLE,
}


def _promote_for_compatible_merge(
    base: Node | None,
    b_kind: ContainerType | None,
    ours: Node | None,
    o_kind: ContainerType | None,
    theirs: Node | None,
    t_kind: ContainerType | None,
) -> tuple[Node, Node, ContainerType] | None:
    """Allow semantically-equivalent structural forms to merge.

    An inline table and a regular table both hold the same mapping; when the
    two sides differ only in that spelling (including when one side still
    uses the base spelling) we merge them as regular tables.
    """
    assert ours is not None and theirs is not None
    assert o_kind is not None and t_kind is not None
    pair: frozenset[ContainerType] = frozenset({o_kind, t_kind})
    common = _PROMOTIONS.get(pair)
    if common is None:
        return None
    if b_kind not in (ContainerType.INLINE_TABLE, ContainerType.TABLE):
        return None
    promoted_ours = (
        _as_table_for_merge(ours)
        if o_kind is ContainerType.INLINE_TABLE
        else ours
    )
    promoted_theirs = (
        _as_table_for_merge(theirs)
        if t_kind is ContainerType.INLINE_TABLE
        else theirs
    )
    return promoted_ours, promoted_theirs, common


def _as_table_for_merge(item: Node) -> Node:
    """View an inline-table / proxy side as a concrete table for merging."""
    if isinstance(item, InlineTable):
        return _expand_inline_table(item)
    return item


def _merge_deletion(
    ctx: _Context,
    path: tuple[str, ...],
    base: Node | None,
    ours: Node | None,
    theirs: Node | None,
    b_kind: ContainerType | None,
    o_kind: ContainerType | None,
    t_kind: ContainerType | None,
) -> Node | None:
    if ours is None and theirs is None:
        return None

    survivor = theirs if ours is None else ours
    survivor_kind = t_kind if ours is None else o_kind
    assert survivor is not None and survivor_kind is not None
    if base is None:
        # One side added what the other side never had: a delete here means
        # "still absent", so the addition wins.
        return _merged_copy(survivor)

    if not _container_touched(base, survivor):
        # The surviving side is identical to base: deletion wins.
        return None

    assert base is not None

    # Deleting a whole map while the other side edits descendants must stay
    # a single whole-path conflict, but an AoT can be merged element-wise:
    # an element deleted here vs. edited there is a conflict on *that
    # element* (and other elements may still merge cleanly).
    if survivor_kind is ContainerType.AOT and isinstance(base, AoT):
        empty = AoT([], parsed=True)
        if ours is None:
            return _merge_aot(ctx, path, base, empty, survivor)
        return _merge_aot(ctx, path, base, survivor, empty)

    if survivor_kind in (ContainerType.TABLE, ContainerType.INLINE_TABLE):
        # Report the conflict once at the container path with per-side
        # evidence; resolving the whole path replaces or removes it.
        kind = ConflictKind.DELETE_DESCENDANTS
        message = (
            "container deleted on one side while descendants were modified"
        )
    else:
        kind = ConflictKind.DELETE
        message = "deleted on one side and modified on the other"

    ctx.report(path, kind, base, ours, theirs, message)
    return _resolved_node(ctx, path, base, ours, theirs)


def _merge_value(
    ctx: _Context,
    path: tuple[str, ...],
    base: Node | None,
    ours: Node | None,
    theirs: Node | None,
) -> Node | None:
    assert ours is not None and theirs is not None

    ours_value_changed = base is None or not _values_equal(base, ours)
    theirs_value_changed = base is None or not _values_equal(base, theirs)
    ours_comment_changed = base is None or _comment_of(base) != _comment_of(ours)
    theirs_comment_changed = (
        base is None or _comment_of(base) != _comment_of(theirs)
    )

    if not ours_value_changed and not theirs_value_changed:
        # Only comments can differ here.
        if _comment_of(ours) == _comment_of(theirs):
            return _merged_copy(ours)
        if not ours_comment_changed:
            return _merged_copy(theirs)
        if not theirs_comment_changed:
            return _merged_copy(ours)
        ctx.report(
            path,
            ConflictKind.COMMENT,
            base,
            ours,
            theirs,
            "the comment was edited on both sides",
        )
        return _resolved_node(ctx, path, base, ours, theirs)

    # At least one side changed the value.
    value_agrees = _values_equal(ours, theirs)
    if value_agrees:
        # Both arrived at the same value; merge comments like a comment-only
        # edit instead of reporting a value conflict.
        if _comment_of(ours) == _comment_of(theirs):
            return _merged_copy(ours)
        if base is not None and not ours_comment_changed:
            return _merged_copy(theirs)
        if base is not None and not theirs_comment_changed:
            return _merged_copy(ours)
        ctx.report(
            path,
            ConflictKind.COMMENT,
            base,
            ours,
            theirs,
            "same new value but the comment was edited on both sides",
        )
        return _resolved_node(ctx, path, base, ours, theirs)

    # Values genuinely differ.
    if not ours_value_changed:
        # Only theirs changed the value.
        result = _merged_copy(theirs)
        if base is not None and ours_comment_changed and not theirs_comment_changed:
            # Ours edited only the comment of the old value.
            if ctx.policy is CommentConflictPolicy.CONFLICT:
                ctx.report(
                    path,
                    ConflictKind.COMMENT,
                    base,
                    ours,
                    theirs,
                    "comment edited on ours side, value changed on theirs",
                )
                return _resolved_node(ctx, path, base, ours, theirs)
            _transfer_comment(result, ours)
        return result

    if not theirs_value_changed:
        result = _merged_copy(ours)
        if (
            base is not None
            and theirs_comment_changed
            and not ours_comment_changed
        ):
            if ctx.policy is CommentConflictPolicy.CONFLICT:
                ctx.report(
                    path,
                    ConflictKind.COMMENT,
                    base,
                    ours,
                    theirs,
                    "value changed on ours side, comment edited on theirs",
                )
                return _resolved_node(ctx, path, base, ours, theirs)
            _transfer_comment(result, theirs)
        return result

    # Both sides changed the value differently: a value conflict.
    ctx.report(
        path,
        ConflictKind.VALUE,
        base,
        ours,
        theirs,
        "the value was changed differently on both sides",
    )
    return _resolved_node(ctx, path, base, ours, theirs)


def _get_child(parent: Node | None, key: str) -> Node | None:
    if parent is None:
        return None
    internal = getattr(parent, "_internal_container", None)
    target: Any = internal if internal is not None else parent
    if not hasattr(target, "item"):
        # A scalar/value node (e.g. an integer base later replaced by a
        # table on both sides) has no children.
        return None
    try:
        value = target.item(key)
    except KeyError:
        return None
    # Container.item() always yields a concrete tomlkit item or an
    # OutOfOrderTableProxy; a bare ``bool`` here means the value was unwrapped
    # via a __getitem__-style fallback, which we do not use for semantics.
    if value is None:
        return None
    return cast(Node, value)


def _merge_map_contents(
    ctx: _Context,
    path: tuple[str, ...],
    base: Node | None,
    ours: Node | None,
    theirs: Node | None,
) -> Node | None:
    """Analyse three same-kind map nodes and build the merged map.

    The rendered map is built by editing a deep copy of *ours* (this is what
    preserves the untouched representation); merged children from theirs are
    copied into it. If ours expresses the node through dotted keys (an
    :class:`OutOfOrderTableProxy`), the copy is built as a plain table from
    the merged children instead of fighting the proxy's layout rules.
    """
    assert ours is not None and theirs is not None

    header_decision = _merge_header_comment(ctx, path, base, ours, theirs)

    if isinstance(ours, OutOfOrderTableProxy):
        merged = _build_map_from_children(
            ctx, path, base, ours, theirs, use_ours_template=False
        )
    else:
        merged = _build_map_from_children(
            ctx, path, base, ours, theirs, use_ours_template=True
        )
    if merged is not None and header_decision is not None:
        merged_header = getattr(merged, "trivia", None)
        if merged_header is not None:
            merged_header.comment = header_decision
    return merged


def _merge_header_comment(
    ctx: _Context,
    path: tuple[str, ...],
    base: Node | None,
    ours: Node,
    theirs: Node,
) -> str | None:
    """Three-way-merge the comment of a table header (None => keep ours)."""
    if not (isinstance(ours, Table) and isinstance(theirs, Table)):
        return None
    ours_comment = _comment_of(ours)
    theirs_comment = _comment_of(theirs)
    if ours_comment == theirs_comment:
        return ours_comment
    base_comment = _comment_of(base) if base is not None else ""
    ours_changed = ours_comment != base_comment
    theirs_changed = theirs_comment != base_comment
    if not ours_changed:
        return theirs_comment
    if not theirs_changed:
        return ours_comment
    ctx.report(
        path,
        ConflictKind.COMMENT,
        base,
        ours,
        theirs,
        "the table header comment was edited on both sides",
    )
    return None


def _build_map_from_children(
    ctx: _Context,
    path: tuple[str, ...],
    base: Node | None,
    ours: Node,
    theirs: Node,
    *,
    use_ours_template: bool,
) -> Node | None:
    keys: dict[str, None] = {}
    for side in (base, ours, theirs):
        if side is not None:
            for key_name in _map_keys(side):
                keys[key_name] = None

    child_results: dict[str, Node | None] = {}
    for key_name in keys:
        child_path = (*path, key_name)
        child_results[key_name] = _merge_node(
            ctx,
            child_path,
            _get_child(base, key_name),
            _get_child(ours, key_name),
            _get_child(theirs, key_name),
        )

    if use_ours_template:
        return _apply_children_to_ours(ours, child_results)
    return _build_plain_map(ours, child_results)


def _apply_children_to_ours(
    ours: Node,
    child_results: dict[str, Node | None],
) -> Node:
    target: Any = copy.deepcopy(ours)

    # An inline table that must host table/AoT children cannot stay inline.
    if isinstance(target, InlineTable) and any(
        isinstance(child, (Table, AoT)) for child in child_results.values()
    ):
        target = _expand_inline_table(target)

    for key_name, child in child_results.items():
        present = key_name in target
        if child is None:
            if present:
                del target[key_name]
            continue
        if present:
            current = _get_child(target, key_name)
            if current is child:
                continue
            target[key_name] = child
        else:
            target[key_name] = child

    return cast(Node, target)


def _build_plain_map(
    ours: Node,
    child_results: dict[str, Node | None],
) -> Node:
    """Build a concrete table from merged children (dotted-key side case)."""
    from tomlkit.items import Trivia

    if isinstance(ours, TOMLDocument):  # pragma: no cover - root is concrete
        target: Any = copy.deepcopy(ours)
    else:
        target = Table(
            Container(),
            copy.copy(getattr(ours, "trivia", Trivia())),
            is_aot_element=False,
            is_super_table=False,
            name=getattr(ours, "name", None),
            display_name=getattr(ours, "display_name", None),
        )

    for key_name, child in child_results.items():
        if child is None:
            continue
        target[key_name] = child
    return target  # type: ignore[no-any-return]


def _expand_inline_table(inline: InlineTable) -> Table:
    """Turn an inline table into a regular ``[table]``-style table."""
    expanded = Table(
        Container(),
        copy.copy(inline.trivia),
        is_aot_element=False,
        is_super_table=False,
        name=getattr(inline, "name", None),
        display_name=getattr(inline, "display_name", None),
    )
    for body_key, body_item in inline.value.body:
        if body_key is None:
            # Inline-only trivia: bare commas and the whitespace around them
            # have no meaning in a regular table; standalone comments are
            # preserved as table-level comments.
            copied_trivia = copy.deepcopy(body_item)
            if isinstance(copied_trivia, Whitespace) and (
                not copied_trivia.s.strip()
                or set(copied_trivia.s) <= {",", " ", "\t", "\r", "\n"}
            ):
                continue
            expanded.raw_append(None, copied_trivia)
            continue
        copied = copy.deepcopy(body_item)
        if not isinstance(copied, (Table, AoT)):
            # Entries parsed inside an inline table use inline separators;
            # a regular table needs newline-terminated, unindented entries.
            copied.trivia.trail = "\n"
            copied.trivia.indent = ""
        expanded.raw_append(copy.deepcopy(body_key), copied)
    expanded.invalidate_display_name()
    return expanded


# ---------------------------------------------------------------------------
# Arrays of tables
# ---------------------------------------------------------------------------


def _aot_identities(
    ctx: _Context,
    path: tuple[str, ...],
    aot: AoT,
) -> list[Hashable | None] | None:
    """Element identities for an AoT according to the caller's policy.

    Returns ``None`` when no policy was provided. Duplicate non-None
    identities are treated as unusable (the merge reports candidates rather
    than guessing which element a key refers to).
    """
    if ctx.identity is None:
        return None
    identities = [ctx.identity(path, element) for element in aot.body]
    seen: dict[Hashable, int] = {}
    for ident in identities:
        if ident is None:
            continue
        seen[ident] = seen.get(ident, 0) + 1
    if any(count > 1 for count in seen.values()):
        ctx.report(
            path,
            ConflictKind.AOT_CANDIDATES,
            None,
            None,
            None,
            "the identity key is not unique on one side",
        )
        return [None] * len(aot.body)
    return identities


def _merge_aot(
    ctx: _Context,
    path: tuple[str, ...],
    base: Node | None,
    ours: Node | None,
    theirs: Node | None,
) -> Node | None:
    assert isinstance(ours, AoT) and isinstance(theirs, AoT)
    base_aot = base if isinstance(base, AoT) else None

    if ctx.resolution_for(path) is not None:
        return _resolved_node(ctx, path, base, ours, theirs)

    o_ids = _aot_identities(ctx, path, ours)
    t_ids = _aot_identities(ctx, path, theirs)
    b_ids = (
        _aot_identities(ctx, path, base_aot) if base_aot is not None else None
    )

    if o_ids is None or t_ids is None or (base_aot is not None and b_ids is None):
        # No usable identity policy: only merge when the array is structurally
        # untouched on both sides (identical length, no add/remove/reorder).
        return _merge_aot_positional(ctx, path, base_aot, ours, theirs)

    return _merge_aot_by_identity(
        ctx, path, base_aot, b_ids or [], ours, o_ids or [], theirs, t_ids or []
    )


def _aot_signature(aot: AoT | None) -> list[Any]:
    if aot is None:
        return []
    return [element.unwrap() for element in aot.body]


def _merge_aot_positional(
    ctx: _Context,
    path: tuple[str, ...],
    base: AoT | None,
    ours: AoT,
    theirs: AoT,
) -> Node | None:
    """Conservative fallback without an identity policy.

    Per-element edits merge positionally only when the array length is
    unchanged on both sides: same length means no element was inserted or
    removed, so index i on each side is plausibly the same element. A pure
    permutation with edits on both sides is still ambiguous, but a length
    change on either side is reported as candidates, since matching added or
    removed elements by position would be a guess.
    """
    base_sig = _aot_signature(base)
    ours_sig = _aot_signature(ours)
    theirs_sig = _aot_signature(theirs)
    base_len = len(base_sig)
    ours_len = len(ours_sig)
    theirs_len = len(theirs_sig)

    if base is None:
        # The AoT was introduced (possibly from a different base type) on
        # both sides. Identical introductions merge; divergent ones cannot be
        # matched by position without an identity policy.
        if _equal_unwrapped(ours_sig, theirs_sig):
            return _merged_copy(ours)
        ctx.report(
            path,
            ConflictKind.AOT_CANDIDATES,
            None,
            ours,
            theirs,
            "array of tables was introduced differently on both sides "
            "without an identity policy",
        )
        return _resolved_node(ctx, path, base, ours, theirs)

    if base_len == ours_len == theirs_len:
        if _looks_like_reorder(base_sig, ours_sig, theirs_sig):
            # The same elements are present but the order moved: positional
            # matching would bind edits to the wrong element, so report
            # candidates instead of guessing.
            ctx.report(
                path,
                ConflictKind.AOT_CANDIDATES,
                base,
                ours,
                theirs,
                "array of tables was reordered without an identity policy",
            )
            return _resolved_node(ctx, path, base, ours, theirs)
        return _merge_aot_elements(
            ctx,
            path,
            list(zip(base.body if base else [], ours.body, theirs.body)),
            mode="index",
        )

    if ours_len == base_len:
        # Only theirs added/removed elements: take theirs wholesale.
        return _merged_copy(theirs)
    if theirs_len == base_len:
        return _merged_copy(ours)

    ctx.report(
        path,
        ConflictKind.AOT_CANDIDATES,
        base,
        ours,
        theirs,
        "array of tables was added to or removed from on both sides "
        "without an identity policy",
    )
    return _resolved_node(ctx, path, base, ours, theirs)



def _looks_like_reorder(
    base_sig: list[Any],
    ours_sig: list[Any],
    theirs_sig: list[Any],
) -> bool:
    """Detect a pure reorder at equal length without an identity.

    When every element on a side still exists in base (as a multiset) but
    the positional sequence differs, the change is consistent with a
    reorder rather than an edit. Only report it when the two sides disagree
    on the order/contents, since an edit on one side alone still merges
    positionally.
    """
    def _is_permutation(changed: list[Any], reference: list[Any]) -> bool:
        if len(changed) != len(reference):
            return False
        remaining = list(reference)
        for element in changed:
            for i, candidate in enumerate(remaining):
                if _equal_unwrapped(element, candidate):
                    del remaining[i]
                    break
            else:
                return False
        return True

    ours_reordered = (
        not _equal_unwrapped(ours_sig, base_sig)
        and _is_permutation(ours_sig, base_sig)
    )
    theirs_reordered = (
        not _equal_unwrapped(theirs_sig, base_sig)
        and _is_permutation(theirs_sig, base_sig)
    )
    if not (ours_reordered or theirs_reordered):
        return False
    return not _equal_unwrapped(ours_sig, theirs_sig)


def _element_path_label(mode: str, ident: Hashable | None, index: int) -> str:
    if mode == "identity":
        return f"@{ident}"
    return str(index)


def _merge_aot_elements(
    ctx: _Context,
    path: tuple[str, ...],
    triples: list[tuple[Item | None, Item, Item]],
    mode: str,
    ids: list[Hashable | None] | None = None,
) -> AoT:
    result = AoT([], parsed=True)
    for index, (b_element, o_element, t_element) in enumerate(triples):
        label = _element_path_label(
            mode, ids[index] if ids is not None else None, index
        )
        element_path = (*path, label)
        merged = _merge_node(
            ctx,
            element_path,
            b_element if isinstance(b_element, Table) else None,
            o_element if isinstance(o_element, Table) else None,
            t_element if isinstance(t_element, Table) else None,
        )
        if isinstance(merged, Table):
            result.body.append(merged)
            list.append(result, merged)
    return result


def _merge_aot_by_identity(
    ctx: _Context,
    path: tuple[str, ...],
    base: AoT | None,
    b_ids: list[Hashable | None],
    ours: AoT,
    o_ids: list[Hashable | None],
    theirs: AoT,
    t_ids: list[Hashable | None],
) -> Node | None:
    """Merge an AoT by stable element identity.

    Elements with ``None`` identity (missing key) cannot be matched, so any
    array containing one is reported as candidates instead of being matched
    by position.
    """
    if any(ident is None for ident in (*o_ids, *t_ids, *b_ids)):
        ctx.report(
            path,
            ConflictKind.AOT_CANDIDATES,
            base,
            ours,
            theirs,
            "at least one element lacks the identity key",
        )
        return _resolved_node(ctx, path, base, ours, theirs)

    base_by_id = (
        dict(zip(b_ids, base.body)) if base is not None else {}
    )
    ours_by_id = dict(zip(o_ids, ours.body))
    theirs_by_id = dict(zip(t_ids, theirs.body))

    all_ids: list[Hashable] = []
    for ident in [*o_ids, *t_ids, *b_ids]:
        if ident not in all_ids:
            all_ids.append(ident)

    result = AoT([], parsed=True)

    for ident in all_ids:
        b_element = base_by_id.get(ident)
        o_element = ours_by_id.get(ident)
        t_element = theirs_by_id.get(ident)

        if o_element is None and t_element is None:
            continue

        if o_element is None or t_element is None:
            survivor = t_element if o_element is None else o_element
            assert survivor is not None
            if ident not in base_by_id:
                # Added only on one side: accept it.
                merged: Node | None = _merged_copy(survivor)
            elif _entry_equal(base_by_id[ident], survivor):
                # Deleted on one side, unchanged on the other: delete.
                merged = None
            else:
                # Delete vs. edit of the same element.
                element_path = (*path, f"@{ident}")
                merged = _merge_node(
                    ctx,
                    element_path,
                    b_element,
                    o_element,
                    t_element,
                )
                if merged is None:
                    continue
        else:
            merged = _merge_node(
                ctx,
                (*path, f"@{ident}"),
                b_element,
                o_element,
                t_element,
            )

        if isinstance(merged, Table):
            result.body.append(merged)
            list.append(result, merged)

    # Output ordering follows ours (so an ours-side reorder is respected);
    # elements present only on theirs are appended afterwards. Because
    # matching is by identity (never by position), any such reorder stays
    # semantically correct instead of binding edits to the wrong element.
    return result


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _as_source(value: str | bytes | TOMLDocument) -> tuple[str, TOMLDocument, dict[int, BoundSpan]]:
    if isinstance(value, TOMLDocument):
        source = value.as_string()
        doc, spans = parse_with_spans(source)
        return source, doc, spans
    if isinstance(value, bytes):
        text = value.decode("utf-8")
    else:
        text = value
    doc, spans = parse_with_spans(text)
    return text, doc, spans


def _compute(
    sources: tuple[str, str, str],
    *,
    policy: CommentConflictPolicy,
    identity: IdentityPolicy | None,
    resolutions: dict[tuple[str, ...], Item | Resolution],
) -> MergeResult:
    def parse_side(name: str, source: str) -> _Side:
        doc, spans = parse_with_spans(source)
        return _Side(name=name, doc=doc, source=source, spans=spans)

    base_source, ours_source, theirs_source = sources
    base_side = parse_side("base", base_source)
    ours_side = parse_side("ours", ours_source)
    theirs_side = parse_side("theirs", theirs_source)

    ctx = _Context(
        sides={
            "base": base_side,
            "ours": ours_side,
            "theirs": theirs_side,
        },
        policy=policy,
        identity=identity,
        resolutions=resolutions,
        conflicts=[],
    )

    merged = _merge_node(
        ctx,
        (),
        base_side.doc,
        ours_side.doc,
        theirs_side.doc,
    )
    if not isinstance(merged, TOMLDocument):  # pragma: no cover - defensive
        raise TOMLKitMergeError("root merge did not produce a document")

    return MergeResult(
        _sources=sources,
        _policy=policy,
        _identity=identity,
        conflicts=ctx.conflicts,
        document=merged,
        _resolutions=dict(resolutions),
    )


def merge3(
    base: str | bytes | TOMLDocument,
    ours: str | bytes | TOMLDocument,
    theirs: str | bytes | TOMLDocument,
    *,
    comment_policy: CommentConflictPolicy = CommentConflictPolicy.MERGE,
    identity: IdentityPolicy | None = None,
) -> MergeResult:
    """Structurally three-way-merge three TOML documents.

    Identity is the semantic TOML path together with the container type:
    ``a.b`` in a dotted key and ``[a]``'s child ``b`` are the same node, and
    an ``[[aot]]`` element is *not* matched by array position. Edits to
    different keys merge automatically; deleting a container while the other
    side edits descendants always conflicts.

    :param base: the common ancestor text/document.
    :param ours: the local side.
    :param theirs: the other side.
    :param comment_policy: what to do when one side changes a value and the
        other side changes only that entry's comment.
    :param identity: optional identity policy for arrays of tables. Pass
        :func:`identity_key("name") <identity_key>` to match elements by a
        stable field. Without it, AoT additions, removals and reorders are
        reported as :attr:`ConflictKind.AOT_CANDIDATES` instead of guessed.
    """
    base_text, _, _ = _as_source(base)
    ours_text, _, _ = _as_source(ours)
    theirs_text, _, _ = _as_source(theirs)
    return _compute(
        (base_text, ours_text, theirs_text),
        policy=comment_policy,
        identity=identity,
        resolutions={},
    )


def merge_documents(
    base: str | bytes | TOMLDocument,
    ours: str | bytes | TOMLDocument,
    theirs: str | bytes | TOMLDocument,
    **kwargs: Any,
) -> TOMLDocument:
    """Like :func:`merge3`, but returns the document and raises on conflicts.

    Use this when a non-clean merge is an error for the caller; otherwise use
    :func:`merge3` and inspect the returned
    :class:`~tomlkit.merge.MergeResult`.
    """
    result = merge3(base, ours, theirs, **kwargs)
    return result.merged_document()
