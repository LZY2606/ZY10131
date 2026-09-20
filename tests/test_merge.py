from __future__ import annotations

from typing import Any

import tomlkit

from tomlkit import CommentPolicy
from tomlkit import Conflict
from tomlkit import ConflictKind
from tomlkit import merge
from tomlkit import parse


def _sem(text: str) -> dict[str, Any]:
    return parse(text).unwrap()


def assert_fragments_preserved(
    ours: str, merged: str, fragments: list[str]
) -> None:
    position = 0
    for fragment in fragments:
        index = merged.find(fragment, position)
        assert index >= position, f"fragment not preserved verbatim: {fragment!r}"
        position = index + len(fragment)


# ---------------------------------------------------------------------------
# Different keys / same key
# ---------------------------------------------------------------------------


def test_different_keys_merge_without_conflict() -> None:
    base = "a = 1\nb = 2\n"
    ours = "a = 1\nb = 3\n"
    theirs = "a = 2\nb = 2\n"

    result = merge(base, ours, theirs)

    assert result.ok
    assert _sem(result.merged.as_string()) == {"a": 2, "b": 3}


def test_same_key_different_values_conflicts_with_evidence() -> None:
    base = "a = 1\n"
    ours = "a = 2\n"
    theirs = "a = 3\n"

    result = merge(base, ours, theirs)

    assert not result.ok
    (conflict,) = result.conflicts
    assert conflict.path == ("a",)
    assert conflict.kind is ConflictKind.VALUE
    assert conflict.base_span is not None
    assert conflict.ours_span is not None
    assert conflict.theirs_span is not None
    assert conflict.base_span.text == "a = 1\n"
    assert conflict.ours_span.text == "a = 2\n"
    assert conflict.theirs_span.text == "a = 3\n"
    assert "a = 2" in result.merged.as_string()


def test_one_side_unchanged_takes_modification() -> None:
    base = "a = 1\nb = 2\n"
    ours = "a = 1\nb = 2\n"
    theirs = "a = 5\nb = 2\n"

    result = merge(base, ours, theirs)

    assert result.ok
    assert _sem(result.merged.as_string()) == {"a": 5, "b": 2}


def test_added_keys_on_each_side_merge() -> None:
    base = "a = 1\n"
    ours = "a = 1\nonly_ours = 7\n"
    theirs = "a = 1\nonly_theirs = 8\n"

    result = merge(base, ours, theirs)

    assert result.ok
    assert _sem(result.merged.as_string()) == {
        "a": 1,
        "only_ours": 7,
        "only_theirs": 8,
    }


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------


def test_comment_only_changes_conflict_by_default() -> None:
    base = "a = 1 # base note\n"
    ours = "a = 1 # ours note\n"
    theirs = "a = 1 # theirs note\n"

    result = merge(base, ours, theirs)

    (conflict,) = result.conflicts
    assert conflict.kind is ConflictKind.COMMENT


def test_value_change_vs_comment_change_policy_conflict() -> None:
    base = "a = 1 # note\n"
    ours = "a = 2 # note\n"
    theirs = "a = 1 # reworded\n"

    result = merge(base, ours, theirs)

    assert len(result.conflicts) == 1
    assert result.conflicts[0].kind is ConflictKind.COMMENT


def test_value_change_vs_comment_change_policy_merge_combines() -> None:
    base = "a = 1 # note\n"
    ours = "a = 2 # note\n"
    theirs = "a = 1 # reworded\n"

    result = merge(base, ours, theirs, comment_policy=CommentPolicy.MERGE)

    assert result.ok
    text = result.merged.as_string()
    assert "a = 2" in text
    assert "# reworded" in text


def test_identical_comment_change_on_both_sides_keeps() -> None:
    base = "a = 1\n"
    ours = "a = 1 # note\n"
    theirs = "a = 1 # note\n"

    result = merge(base, ours, theirs)
    assert result.ok


# ---------------------------------------------------------------------------
# Deletion vs modification
# ---------------------------------------------------------------------------


def test_delete_leaf_while_other_side_modifies_conflicts() -> None:
    base = "a = 1\n"
    ours = ""
    theirs = "a = 2\n"

    result = merge(base, ours, theirs)

    (conflict,) = result.conflicts
    assert conflict.kind is ConflictKind.DELETE_MODIFIED
    assert conflict.path == ("a",)


def test_delete_container_while_descendant_modified_conflicts() -> None:
    base = "[t]\nx = 1\n[t.sub]\ny = 2\n"
    ours = "[t]\nx = 1\n"
    theirs = "[t]\nx = 1\n[t.sub]\ny = 9\n"

    result = merge(base, ours, theirs)

    kinds = {c.kind for c in result.conflicts}
    assert ConflictKind.DELETE_DESCENDANT in kinds
    path = next(c.path for c in result.conflicts if c.kind is ConflictKind.DELETE_DESCENDANT)
    assert path == ("t", "sub")


def test_delete_container_when_other_side_unchanged_wins() -> None:
    base = "[t]\nx = 1\nother = 2\n"
    ours = "other = 2\n"
    theirs = "[t]\nx = 1\nother = 2\n"

    result = merge(base, ours, theirs)

    assert result.ok
    assert _sem(result.merged.as_string()) == {"other": 2}


# ---------------------------------------------------------------------------
# Dotted keys and explicit tables resolve to the same path
# ---------------------------------------------------------------------------


def test_dotted_key_and_explicit_table_share_path() -> None:
    base = "a.b = 1\n"
    ours = "a.b = 2\n"
    theirs = "[a]\nb = 1\nc = 3\n"

    result = merge(base, ours, theirs)

    assert result.ok
    assert _sem(result.merged.as_string()) == {"a": {"b": 2, "c": 3}}


def test_explicit_table_on_ours_and_dotted_on_theirs() -> None:
    base = "[a]\nb = 1\n"
    ours = "[a]\nb = 2\n"
    theirs = "a.b = 1\na.c = 5\n"

    result = merge(base, ours, theirs)

    assert result.ok
    assert _sem(result.merged.as_string()) == {"a": {"b": 2, "c": 5}}


def test_no_duplicate_or_out_of_order_tables() -> None:
    base = "[a]\nb = 1\n[a.c]\nd = 1\n"
    ours = "[a]\nb = 2\n[a.c]\nd = 1\n"
    theirs = "[a]\nb = 1\n[a.c]\nd = 9\ne = 4\n"

    result = merge(base, ours, theirs)
    text = result.merged.as_string()

    assert result.ok
    # A structural rebuild must never render the same header twice.
    assert text.count("[a.c]") == 1
    assert _sem(text) == {"a": {"b": 2, "c": {"d": 9, "e": 4}}}


# ---------------------------------------------------------------------------
# Array of tables
# ---------------------------------------------------------------------------


AOT_BASE = (
    "[[s]]\n"
    'name = "one"\n'
    "port = 1\n"
    "\n"
    "[[s]]\n"
    'name = "two"\n'
    "port = 2\n"
)


def test_aot_different_fields_merge_with_identity() -> None:
    ours = AOT_BASE.replace("port = 1", "port = 11")
    theirs = AOT_BASE.replace("port = 2", "port = 22")

    result = merge(AOT_BASE, ours, theirs, identity={("s",): "name"})

    assert result.ok
    assert _sem(result.merged.as_string()) == {
        "s": [
            {"name": "one", "port": 11},
            {"name": "two", "port": 22},
        ]
    }


def test_aot_add_on_one_side_merges() -> None:
    added = AOT_BASE + "\n[[s]]\n" + 'name = "three"\n'

    result = merge(AOT_BASE, AOT_BASE, added, identity={("s",): "name"})

    assert result.ok
    assert [e["name"] for e in _sem(result.merged.as_string())["s"]] == [
        "one",
        "two",
        "three",
    ]


def test_aot_delete_and_modify_different_elements_merges() -> None:
    # ours deletes "two"; theirs modifies "one" -> different identities.
    ours = "[[s]]\n" + 'name = "one"\n' + "port = 1\n"
    theirs = (
        "[[s]]\n"
        'name = "one"\n'
        "port = 99\n"
        "\n"
        "[[s]]\n"
        'name = "two"\n'
        "port = 2\n"
    )

    result = merge(AOT_BASE, ours, theirs, identity={("s",): "name"})

    assert result.ok
    assert _sem(result.merged.as_string()) == {"s": [{"name": "one", "port": 99}]}


def test_aot_delete_same_element_that_other_side_modifies_conflicts() -> None:
    ours = "[[s]]\n" + 'name = "two"\n' + "port = 2\n"
    theirs = (
        "[[s]]\n"
        'name = "one"\n'
        "port = 99\n"
        "\n"
        "[[s]]\n"
        'name = "two"\n'
        "port = 2\n"
    )

    result = merge(AOT_BASE, ours, theirs, identity={("s",): "name"})

    kinds = {c.kind for c in result.conflicts}
    assert kinds <= {ConflictKind.DELETE_MODIFIED, ConflictKind.DELETE_DESCENDANT}
    assert kinds


def test_aot_without_identity_reports_candidates_instead_of_guessing() -> None:
    base = "[[s]]\nx = 1\n"
    ours = "[[s]]\nx = 2\n"
    theirs = "[[s]]\nx = 3\n"

    result = merge(base, ours, theirs)

    (conflict,) = result.conflicts
    assert conflict.kind is ConflictKind.AOT_AMBIGUOUS
    assert conflict.path == ("s",)
    assert conflict.candidates  # conservative evidence, no positional guess


def test_aot_without_identity_one_sided_add_still_merges() -> None:
    base = "[[s]]\nx = 1\n"
    added = "[[s]]\nx = 1\n\n[[s]]\nx = 9\n"

    result = merge(base, base, added)

    assert result.ok
    assert _sem(result.merged.as_string()) == {"s": [{"x": 1}, {"x": 9}]}


def test_aot_untouched_element_kept_verbatim() -> None:
    base = (
        "[[s]]\n"
        'name = "one"\n'
        "port = 1\n"
        "\n"
        "[[s]]\n"
        'name = "two"\n'
        "port = 2\n"
    )
    theirs = (
        "[[s]]\n"
        'name = "one"\n'
        "port = 99\n"
        "\n"
        "[[s]]\n"
        'name = "two"\n'
        "port = 2\n"
    )

    result = merge(base, base, theirs, identity={("s",): "name"})
    merged = result.merged.as_string()

    assert_fragments_preserved(
        base, merged, ['[[s]]\nname = "two"\nport = 2\n']
    )


def test_aot_callable_identity_key() -> None:
    base = AOT_BASE
    theirs = (
        "[[s]]\n"
        'name = "one"\n'
        "port = 1\n"
        "extra = 5\n"
        "\n"
        "[[s]]\n"
        'name = "two"\n'
        "port = 2\n"
    )

    result = merge(
        base, base, theirs, identity={("s",): lambda element: element["name"]}
    )

    assert result.ok
    assert _sem(result.merged.as_string()) == {
        "s": [
            {"name": "one", "port": 1, "extra": 5},
            {"name": "two", "port": 2},
        ]
    }


def test_aot_one_side_adds_other_reorders_merges() -> None:
    # Only one side reorders (and the other side only appends): alignable.
    ours_added = AOT_BASE + "\n[[s]]\n" + 'name = "three"\n'
    theirs_reversed = (
        "[[s]]\n"
        'name = "two"\n'
        "port = 2\n"
        "\n"
        "[[s]]\n"
        'name = "one"\n'
        "port = 1\n"
    )

    result = merge(
        AOT_BASE, ours_added, theirs_reversed, identity={("s",): "name"}
    )

    assert result.ok
    assert [e["name"] for e in _sem(result.merged.as_string())["s"]] == [
        "two",
        "one",
        "three",
    ]


def test_aot_both_sides_reorder_differently_conflicts() -> None:
    ours_reversed = (
        "[[s]]\n"
        'name = "two"\n'
        "port = 2\n"
        "\n"
        "[[s]]\n"
        'name = "one"\n'
        "port = 1\n"
    )
    # Theirs keeps base order but changes a value (a different, conflicting
    # intention). To make it a genuine two-way reorder, use a third element
    # that both sides arrange differently.
    base3 = AOT_BASE + "\n[[s]]\n" + 'name = "three"\n' + "port = 3\n"
    ours_order = (
        "[[s]]\n"
        'name = "three"\n'
        "port = 3\n"
        "\n"
        "[[s]]\n"
        'name = "one"\n'
        "port = 1\n"
        "\n"
        "[[s]]\n"
        'name = "two"\n'
        "port = 2\n"
    )
    theirs_order = (
        "[[s]]\n"
        'name = "two"\n'
        "port = 2\n"
        "\n"
        "[[s]]\n"
        'name = "one"\n'
        "port = 1\n"
        "\n"
        "[[s]]\n"
        'name = "three"\n'
        "port = 3\n"
    )

    result = merge(base3, ours_order, theirs_order, identity={("s",): "name"})

    assert any(c.kind is ConflictKind.AOT_REORDER for c in result.conflicts)


# ---------------------------------------------------------------------------
# Inline tables and type changes
# ---------------------------------------------------------------------------


def test_inline_table_internal_changes_merge() -> None:
    base = "p = {x = 1, y = 2}\n"
    ours = "p = {x = 9, y = 2}\n"
    theirs = "p = {x = 1, y = 8}\n"

    result = merge(base, ours, theirs)

    assert result.ok
    assert _sem(result.merged.as_string()) == {"p": {"x": 9, "y": 8}}


def test_inline_to_table_is_structure_conflict() -> None:
    base = "p = {x = 1}\n"
    ours = "p = {x = 1}\n"
    theirs = "[p]\nx = 2\n"

    result = merge(base, ours, theirs)

    (conflict,) = result.conflicts
    assert conflict.kind is ConflictKind.STRUCTURE


def test_date_vs_string_is_value_conflict_not_structure() -> None:
    base = "d = 2020-01-01\n"
    ours = "d = 2020-02-02\n"
    theirs = 'd = "not a date"\n'

    result = merge(base, ours, theirs)

    (conflict,) = result.conflicts
    assert conflict.kind is ConflictKind.VALUE


def test_string_style_difference_same_value_keeps_ours() -> None:
    base = 's = "hi"\n'
    ours = "s = 'hi'\n"
    theirs = 's = "hi"\n'

    result = merge(base, ours, theirs)

    assert result.ok
    assert result.merged.as_string() == "s = 'hi'\n"


# ---------------------------------------------------------------------------
# CRLF
# ---------------------------------------------------------------------------


def test_crlf_line_endings_preserved_in_untouched_and_changed_lines() -> None:
    base = "a = 1\r\nb = 2\r\n"
    ours = "a = 1\r\nb = 3\r\n"
    theirs = "a = 2\r\nb = 2\r\n"

    result = merge(base, ours, theirs)

    assert result.ok
    text = result.merged.as_string()
    assert text == "a = 2\r\nb = 3\r\n"
    assert "\n" not in text.replace("\r\n", "")


# ---------------------------------------------------------------------------
# Untouched regions and structural reapplication
# ---------------------------------------------------------------------------


def test_untouched_region_keeps_ours_bytes_verbatim() -> None:
    base = (
        "# header comment\n"
        'title = "base"\n'
        "\n"
        "[server]\n"
        'host = "localhost" # trailing note\n'
        "port = 8080\n"
        "\n"
        "[untouched]\n"
        "# a preserved note\n"
        "deep.value = 42\n"
        "keep = true\n"
    )
    ours = base.replace('title = "base"', 'title = "ours-title"')
    theirs = (
        "# header comment\n"
        'title = "base"\n'
        "\n"
        "[server]\n"
        'host = "example.com"\n'
        "port = 9090\n"
        "\n"
        "[untouched]\n"
        "# a preserved note\n"
        "deep.value = 42\n"
        "keep = true\n"
    )

    result = merge(base, ours, theirs)
    merged = result.merged.as_string()

    assert_fragments_preserved(
        ours,
        merged,
        [
            "# header comment\n",
            "[untouched]\n# a preserved note\ndeep.value = 42\nkeep = true\n",
        ],
    )
    # Merged content is semantically the union of both edits.
    assert _sem(merged) == {
        "title": "ours-title",
        "server": {"host": "example.com", "port": 9090},
        "untouched": {"deep": {"value": 42}, "keep": True},
    }


def test_resolution_take_theirs_and_custom_item_recomputes_container() -> None:
    base = 'a = 1\n[p]\nx = 1\n'
    ours = 'a = 2\n[p]\nx = 9\n'
    theirs = 'a = 3\n[p]\nx = 1\ny = 5\n'

    result = merge(base, ours, theirs)
    assert len(result.conflicts) == 1

    resolved_theirs = result.apply({("a",): "theirs"})
    assert _sem(resolved_theirs.as_string()) == {
        "a": 3,
        "p": {"x": 9, "y": 5},
    }

    resolved_custom = result.apply({("a",): tomlkit.integer(99)})
    assert _sem(resolved_custom.as_string()) == {
        "a": 99,
        "p": {"x": 9, "y": 5},
    }


def test_resolution_callable_and_delete() -> None:
    base = 'a = 1\n'
    ours = 'a = 2\n'
    theirs = 'a = 3\n'

    result = merge(base, ours, theirs)

    assert _sem(result.apply(lambda c: "theirs").as_string()) == {"a": 3}
    assert _sem(result.apply(lambda c: "delete").as_string()) == {}
    # Default (no matching resolution) conservatively keeps ours.
    assert _sem(result.apply({}).as_string()) == {"a": 2}


def test_structure_conflict_resolved_with_custom_item() -> None:
    base = "p = {x = 1}\n"
    ours = "p = {x = 1}\n"
    theirs = "[p]\nx = 2\n"

    result = merge(base, ours, theirs)
    (conflict,) = result.conflicts
    assert conflict.kind is ConflictKind.STRUCTURE

    replacement = tomlkit.inline_table()
    replacement["z"] = 7
    resolved = result.apply({("p",): replacement})

    assert _sem(resolved.as_string()) == {"p": {"z": 7}}


def test_conflict_carries_values_and_three_spans() -> None:
    base = 'a = 1\n'
    ours = 'a = 2\n'
    theirs = 'a = 3\n'

    result = merge(base, ours, theirs)
    conflict: Conflict = result.conflicts[0]

    assert conflict.base is not None
    assert conflict.ours is not None
    assert conflict.theirs is not None
    assert conflict.base.unwrap() == 1
    assert conflict.ours.unwrap() == 2
    assert conflict.theirs.unwrap() == 3
    assert conflict.base_span is not None
    assert conflict.ours_span is not None
    assert conflict.theirs_span is not None
    assert conflict.base_span.start == conflict.ours_span.start
