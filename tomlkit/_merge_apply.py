"""Document construction for the structured three-way merge.

The build always starts from a deep copy of *ours* and applies the recorded
structural decisions in semantic path order. Replacing or deleting a key
recomputes the containing :class:`~tomlkit.container.Container` through
tomlkit's own mutation API, so the output can never contain duplicate tables,
out-of-order headers, or swallowed sibling trivia. Regions the merge does not
touch keep *ours*' exact parsed representation.
"""

from __future__ import annotations

import copy

from typing import TYPE_CHECKING
from typing import Any

from tomlkit.items import AoT
from tomlkit.items import InlineTable
from tomlkit.items import Item
from tomlkit.items import Table
from tomlkit.merge import Conflict
from tomlkit.merge import ConflictKind
from tomlkit.merge import _Action
from tomlkit.merge import _ElementDecision
from tomlkit.merge import _Merger


if TYPE_CHECKING:
    from collections.abc import Callable

    from tomlkit.container import Container
    from tomlkit.container import OutOfOrderTableProxy
    from tomlkit.toml_document import TOMLDocument


MutableContainer = Any  # Container | OutOfOrderTableProxy, both mapping-like

KEEP_OURS = "ours"
TAKE_THEIRS = "theirs"
DELETE_RESOLUTION = "delete"


def build_initial_document(
    ours_doc: TOMLDocument,
    theirs_doc: TOMLDocument,
    merger: _Merger,
) -> tuple[TOMLDocument, list[tuple[str, ...]]]:
    builder = _Builder(merger)
    return builder.build(ours_doc, theirs_doc), builder.touched


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class _Builder:
    def __init__(self, merger: _Merger) -> None:
        self.merger = merger
        self.touched: list[tuple[str, ...]] = []
        # Maps an AoT element logical path (aot_path, index) -> live Table in
        # the document being built, populated while each AoT body is rebuilt.
        self.element_tables: dict[tuple[tuple[str, ...], int], Table] = {}

    def build(
        self, ours_doc: TOMLDocument, theirs_doc: TOMLDocument
    ) -> TOMLDocument:
        self.doc = copy.deepcopy(ours_doc)
        self.theirs_doc = theirs_doc
        self._apply_table(())
        return self.doc

    # -- path navigation ----------------------------------------------------

    def _live_container(self, path: tuple[str, ...]) -> MutableContainer:
        """Resolve a logical table path to the live Container to mutate.

        A numeric component selects an AoT element table; the live element
        tables are registered in ``element_tables`` when the parent AoT body
        is rebuilt.
        """
        container: MutableContainer = self.doc
        cur: tuple[str, ...] = ()
        parts = list(path)
        while parts:
            part = parts.pop(0)
            cur = (*cur, part)
            value = _container_item(container, part)
            if isinstance(value, AoT):
                if not parts:
                    msg = f"expected element index after AoT {'.'.join(cur)}"
                    raise TypeError(msg)
                index_part = parts.pop(0)
                cur = (*cur, index_part)
                table = self.element_tables[(cur[:-1], int(index_part))]
                container = table.value
                continue
            if isinstance(value, (Table, InlineTable)):
                container = value.value
                continue
            from tomlkit.container import OutOfOrderTableProxy

            if isinstance(value, OutOfOrderTableProxy):
                container = value
                continue
            msg = f"expected table at {'.'.join(cur)} (got {type(value).__name__})"
            raise TypeError(msg)
        return container

    def _apply_table(self, path: tuple[str, ...]) -> None:
        ops = self.merger.table_ops.get(path, {})
        container = self._live_container(path)

        for name, decision in ops.items():
            if name == "__header__":
                continue
            self._apply_child(container, (*path, name), decision)

    def _apply_child(
        self, container: Container, path: tuple[str, ...], decision: Any
    ) -> None:
        action = decision.action
        name = path[-1]

        if action is _Action.KEEP:
            return
        if action is _Action.MERGE_TABLE:
            if path in self.merger.aot_plans:
                self._apply_aot(container, name, path)
            else:
                self._apply_table(path)
            return
        if action is _Action.MERGE_AOT:
            self._apply_aot(container, name, path)
            return
        if action is _Action.CONFLICT:
            return

        self.touched.append(path)

        if action is _Action.DELETE:
            if name in container:
                del container[name]
            return

        if action is _Action.TAKE_THEIRS:
            value = copy.deepcopy(decision.take_item)
            self._assign(container, name, value)
            self._apply_inserted_subtree(path, value)
            return

        if action is _Action.COMBINE:
            value = copy.deepcopy(decision.take_item)
            comment_item = decision.comment_item
            if comment_item is not None:
                value.trivia.comment_ws = comment_item.trivia.comment_ws or " "
                value.trivia.comment = comment_item.trivia.comment
            self._assign(container, name, value)
            return

    def _apply_inserted_subtree(self, path: tuple[str, ...], value: Item) -> None:
        if isinstance(value, (Table, InlineTable)):
            self._apply_table(path)

    @staticmethod
    def _assign(container: MutableContainer, name: str, value: Item) -> None:
        if name in container:
            container[name] = value
        else:
            append = getattr(container, "append", None)
            if append is None:
                container[name] = value
                return
            container.append(name, value)

    # -- AoT -----------------------------------------------------------------

    def _apply_aot(self, container: Container, name: str, path: tuple[str, ...]) -> None:
        plan = self.merger.aot_plans.get(path)
        if plan is None or name not in container:
            return
        current = container.item(name)
        if not isinstance(current, AoT):
            return

        built = self._build_aot_body(current, path, plan)
        if built[0] is None:
            return
        body, element_map = built
        assert element_map is not None

        # ``del current[:]`` runs AoT.__delitem__, which clears *both* the
        # underlying list and ``_body`` -- so only clear the list view first,
        # then point ``_body`` at the freshly built body and repopulate.
        final_body = list(body)
        list.__delitem__(current, slice(None, None, None))
        current._body = final_body
        list.extend(current, final_body)
        # Register element tables, then apply child ops in path order.
        for (aot_path, index), table in element_map.items():
            self.element_tables[(aot_path, index)] = table
        applied_paths: set[tuple[str, ...]] = set()
        for _aot_path, index in element_map:
            element_path = (*path, str(index))
            if element_path in applied_paths:
                continue
            applied_paths.add(element_path)
            self._apply_table(element_path)
        self.touched.append(path)

    def _build_aot_body(
        self,
        current: AoT,
        path: tuple[str, ...],
        plan: list[_ElementDecision],
    ) -> tuple[list[Table], dict[tuple[tuple[str, ...], int], Table]] | tuple[None, None]:
        body: list[Table] = []
        element_map: dict[tuple[tuple[str, ...], int], Table] = {}
        ours_by_index = {i: table for i, table in enumerate(current.body)}

        for slot, decision in enumerate(plan):
            table = self._build_element(current, decision, ours_by_index)
            if table is _SKIP:
                continue
            if table is _CONFLICT:
                if decision.ours_index is not None:
                    original = ours_by_index.get(decision.ours_index)
                    if original is not None:
                        body.append(original)
                        element_map[(path, slot)] = original
                continue
            assert isinstance(table, Table)
            body.append(table)
            # Logical index for child-op lookup: use ours index when present so
            # recorded (aot, index) paths resolve, otherwise the slot.
            logical = decision.ours_index if decision.ours_index is not None else slot
            element_map[(path, logical)] = table

        self._normalize_body_indents(current, body)
        return body, element_map

    def _normalize_body_indents(self, current: AoT, body: list[Table]) -> None:
        """Normalize inter-element spacing.

        Parsed AoT elements carry no leading newline in ``indent``; the
        separation lives in the previous element's trailing whitespace. A
        cloned theirs element that was last in its source keeps a blank-line
        trail, so collapse every non-final separator to a single newline and
        let the final element keep its trail.
        """
        for position, table in enumerate(body):
            if position == 0:
                table.trivia.indent = table.trivia.indent.lstrip("\r\n")
            else:
                stripped = table.trivia.indent.lstrip("\r\n")
                table.trivia.indent = "\n" + stripped
            if not table.trivia.trail:
                table.trivia.trail = "\n"
            if position < len(body) - 1:
                self._trim_trailing_blank_lines(table)

    @staticmethod
    def _trim_trailing_blank_lines(table: Table) -> None:
        """Remove whitespace-only body fragments at the end of an element.

        Those fragments carried the blank line that separated this element
        from its original successor; after a rebuild the successor's own
        indent provides the separator, so keeping both doubles the gap.
        """
        from tomlkit.items import Whitespace

        contained = table.value.body
        while contained:
            key, value = contained[-1]
            if key is not None or not isinstance(value, Whitespace):
                break
            contained.pop()
        # Collapse any multi-newline trail on the last real item to a single
        # newline (the successor indent adds the separation).
        for key, value in reversed(contained):
            if key is not None:
                trail = value.trivia.trail
                if "\n" in trail:
                    value.trivia.trail = "\n"
                break

    def _build_element(
        self,
        current: AoT,
        decision: _ElementDecision,
        ours_by_index: dict[int, Table],
    ) -> Table | str:
        action = decision.action
        if action is _Action.DELETE:
            return _SKIP
        if action is _Action.KEEP:
            if decision.ours_index is None:
                return _SKIP
            return ours_by_index[decision.ours_index]
        if action is _Action.CONFLICT:
            return _CONFLICT
        if action is _Action.TAKE_THEIRS:
            assert decision.node is not None
            table = copy.deepcopy(decision.node.item)
            assert isinstance(table, Table)
            table._is_aot_element = True
            return table
        if action is _Action.MERGE_TABLE:
            if decision.ours_index is None:
                return _CONFLICT
            return ours_by_index[decision.ours_index]
        return _CONFLICT



_SKIP = "__skip__"
_CONFLICT = "__conflict__"


def _container_item(container: Container, key: str) -> Item | None:
    from tomlkit.items import SingleKey

    single = SingleKey(key)
    if single not in container:
        return None
    getter = getattr(container, "item", None)
    if getter is not None:
        value: Item | None = getter(key)
        return value
    # OutOfOrderTableProxy exposes the merged value via __getitem__.
    return container[key]  # type: ignore[no-any-return]



# ---------------------------------------------------------------------------
# Conflict resolution
# ---------------------------------------------------------------------------


def build_resolved_document(
    result: Any,
    resolution: "Callable[[Conflict], object] | dict[tuple[str, ...], object] | None",
) -> "TOMLDocument":
    from tomlkit.merge import merge as structured_merge

    docs = result._input_docs
    fresh = structured_merge(
        docs["base"],
        docs["ours"],
        docs["theirs"],
        comment_policy=result._comment_policy,
        identity=result._identity,
    )
    return _Resolver(fresh, result, resolution).build()


class _Resolver:
    """Re-applies a merge with explicit per-conflict resolutions.

    A resolution is ``"ours"``, ``"theirs"``, ``"delete"``, or an
    :class:`~tomlkit.items.Item`/plain value used as the replacement. After
    overriding decisions, the affected containers are recomputed through the
    same structural builder as the unresolved merge.
    """

    def __init__(
        self,
        fresh: Any,
        original: Any,
        resolution: "Callable[[Conflict], object] | dict[tuple[str, ...], object] | None",
    ) -> None:
        self.result = fresh
        self.original = original
        self.resolution = resolution or {}

    def _resolve_for(self, conflict: Conflict) -> object:
        if callable(self.resolution):
            return self.resolution(conflict)
        return self.resolution.get(conflict.path, KEEP_OURS)

    def build(self) -> "TOMLDocument":
        from tomlkit.items import item as make_item
        from tomlkit.merge import _Action
        from tomlkit.merge import _Decision
        from tomlkit.merge import _ElementDecision

        merger = self.result._merger
        overrides: dict[int, object] = {}
        for conflict in self.result.conflicts:
            overrides[id(conflict)] = self._resolve_for(conflict)

        self._apply_overrides(merger, overrides)

        builder = _Builder(merger)
        return builder.build(
            self.result._input_docs["ours"],
            self.result._input_docs["theirs"],
        )

    def _apply_overrides(self, merger: Any, overrides: dict[int, object]) -> None:
        from tomlkit.items import item as make_item
        from tomlkit.merge import _Action
        from tomlkit.merge import _Decision
        from tomlkit.merge import _ElementDecision

        for conflict in self.result.conflicts:
            choice = overrides[id(conflict)]
            self._override_one(merger, conflict, choice)

    def _override_one(self, merger: Any, conflict: Conflict, choice: object) -> None:
        from tomlkit.items import item as make_item
        from tomlkit.merge import _Action
        from tomlkit.merge import _Decision
        from tomlkit.merge import _ElementDecision

        path = conflict.path
        action, value = self._choice_to_action(conflict, choice)

        if conflict.kind in (ConflictKind.AOT_AMBIGUOUS, ConflictKind.AOT_REORDER):
            self._override_aot(merger, conflict, action, value)
            return

        parent = path[:-1]
        name = path[-1]
        container_ops = merger.table_ops.setdefault(parent, {})
        container_ops[name] = _Decision(action, take_item=value)

    def _choice_to_action(
        self, conflict: Conflict, choice: object
    ) -> tuple[Any, Item | Any]:
        from tomlkit.items import item as make_item
        from tomlkit.merge import _Action

        if isinstance(choice, str):
            normalized = choice.strip().lower()
            if normalized in (KEEP_OURS, "keep_ours", "our"):
                return _Action.KEEP, None
            if normalized in (TAKE_THEIRS, "their", "theirs"):
                return _Action.TAKE_THEIRS, conflict.theirs
            if normalized in (DELETE_RESOLUTION, "remove"):
                return _Action.DELETE, None
        if isinstance(choice, Item):
            return _Action.TAKE_THEIRS, choice
        if choice is not None and not isinstance(choice, str):
            return _Action.TAKE_THEIRS, make_item(choice)
        # Unknown string: conservatively keep ours.
        return _Action.KEEP, None

    def _override_aot(
        self, merger: Any, conflict: Conflict, action: Any, value: Item | None
    ) -> None:
        from tomlkit.merge import _Action
        from tomlkit.merge import _Decision

        path = conflict.path
        parent = path[:-1]
        name = path[-1]
        if action is _Action.KEEP:
            # Keep ours AoT body entirely; drop element plan.
            merger.aot_plans[path] = []
            merger.table_ops.setdefault(parent, {})[name] = _Decision(
                _Action.MERGE_AOT
            )
            return
        if action is _Action.DELETE:
            merger.aot_plans[path] = []
            merger.table_ops.setdefault(parent, {})[name] = _Decision(
                _Action.DELETE
            )
            return
        # Take theirs (or a custom replacement Item) for the whole AoT.
        replacement = value if value is not None else conflict.theirs
        merger.aot_plans[path] = []
        merger.table_ops.setdefault(parent, {})[name] = _Decision(
            _Action.TAKE_THEIRS, take_item=replacement
        )
