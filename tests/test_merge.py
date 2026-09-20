from __future__ import annotations

import pytest

import tomlkit

from tomlkit.api import integer
from tomlkit.merge import CommentConflictPolicy
from tomlkit.merge import Conflict
from tomlkit.merge import ConflictKind
from tomlkit.merge import MergeResult
from tomlkit.merge import Resolution
from tomlkit.merge import TOMLKitMergeError
from tomlkit.merge import identity_key
from tomlkit.merge import merge3
from tomlkit.merge import merge_documents


def _semantics(result: MergeResult) -> dict[str, object]:
    return tomlkit.parse(result.document.as_string()).unwrap()


def test_independent_key_edits_merge_and_roundtrip() -> None:
    base = '[server]\nhost = "localhost"\nport = 8000\n'
    ours = '[server]\nhost = "localhost"\nport = 9000\n'
    theirs = '[server]\nhost = "example.com"\nport = 8000\n'

    result = merge3(base, ours, theirs)

    assert result.clean
    assert _semantics(result) == {
        "server": {"host": "example.com", "port": 9000}
    }


def test_comment_only_edit_is_kept_when_other_side_changes_other_key() -> None:
    base = 'a = 1 # note\nb = 2\n'
    ours = 'a = 1 # our note\nb = 2\n'
    theirs = 'a = 1 # note\nb = 3\n'

    result = merge3(base, ours, theirs)

    assert result.clean
    assert result.document.as_string() == 'a = 1 # our note\nb = 3\n'


def test_divergent_comment_edits_conflict() -> None:
    result = merge3('a = 1 # base\n', 'a = 1 # ours\n', 'a = 1 # theirs\n')

    assert not result.clean
    (conflict,) = result.conflicts
    assert conflict.path == ("a",)
    assert conflict.kind is ConflictKind.COMMENT
    assert conflict.base_span is not None
    assert conflict.base_span.text == "a = 1 # base\n"


def test_value_change_and_comment_change_policy() -> None:
    base = 'a = 1\n'
    ours = 'a = 2\n'
    theirs = 'a = 1 # why\n'

    merged = merge3(base, ours, theirs)
    assert merged.clean
    assert merged.document.as_string() == 'a = 2 # why\n'

    conflicting = merge3(
        base, ours, theirs, comment_policy=CommentConflictPolicy.CONFLICT
    )
    assert [c.kind for c in conflicting.conflicts] == [ConflictKind.COMMENT]


def test_delete_container_vs_descendant_edit_conflicts() -> None:
    base = '[t]\nx = 1\n'
    result = merge3(base, '', '[t]\nx = 2\n')

    assert not result.clean
    (conflict,) = result.conflicts
    assert conflict.path == ("t",)
    assert conflict.kind is ConflictKind.DELETE_DESCENDANTS
    assert conflict.ours_value is None
    assert conflict.theirs_value is not None


def test_delete_scalar_vs_edit_is_delete_conflict() -> None:
    result = merge3('a = 1\n', '', 'a = 2\n')

    (conflict,) = result.conflicts
    assert conflict.path == ("a",)
    assert conflict.kind is ConflictKind.DELETE


def test_dotted_key_and_explicit_table_are_same_path() -> None:
    base = '[a]\nb = 1\nc = 2\n'
    ours = 'a.b = 11\na.c = 2\n'
    theirs = '[a]\nb = 1\nc = 22\n'

    result = merge3(base, ours, theirs)

    assert result.clean
    assert _semantics(result) == {"a": {"b": 11, "c": 22}}


def test_string_vs_date_is_value_conflict() -> None:
    result = merge3('k = 1\n', 'k = "2020-01-01"\n', 'k = 2020-01-01\n')

    (conflict,) = result.conflicts
    assert conflict.kind is ConflictKind.VALUE
    assert conflict.path == ("k",)


def test_inline_table_vs_scalar_is_structure_conflict() -> None:
    result = merge3('k = 1\n', 'k = {x = 1}\n', 'k = "s"\n')

    (conflict,) = result.conflicts
    assert conflict.kind is ConflictKind.STRUCTURE


def test_table_vs_scalar_is_structure_conflict() -> None:
    result = merge3('k = {x = 1}\n', '[k]\nx = 1\ny = 2\n', 'k = 1\n')

    (conflict,) = result.conflicts
    assert conflict.kind is ConflictKind.STRUCTURE


def test_resolve_with_builtin_sides_and_custom_item() -> None:
    result = merge3('a = 1\n', 'a = 2\n', 'a = 3\n')
    assert not result.clean

    theirs = result.resolve({"a": Resolution.THEIRS})
    assert theirs.clean
    assert theirs.document.as_string() == 'a = 3\n'

    custom = result.resolve({"a": integer(42)})
    assert custom.clean
    assert custom.document.as_string() == 'a = 42\n'

    with_dotted_path = result.resolve(a=Resolution.OURS)
    assert with_dotted_path.document.as_string() == 'a = 2\n'


def test_conflict_carries_three_source_spans() -> None:
    result = merge3(
        'a = 1\n', 'a = 2  # two\n', 'a = 3\n'
    )
    (conflict,) = result.conflicts

    assert conflict.ours_span is not None
    assert conflict.base_span is not None
    assert conflict.theirs_span is not None
    assert conflict.ours_span.text == 'a = 2  # two\n'
    assert conflict.base_span.text == 'a = 1\n'
    assert conflict.theirs_span.text == 'a = 3\n'
    assert conflict.ours_span.start_line == 1
    assert isinstance(conflict, Conflict)


def test_implicit_super_table_has_no_span_but_path_still_conflicts() -> None:
    base = 'a.b = 1\n'
    result = merge3(base, 'a.b = 2\n', 'a.b = 3\n')

    (conflict,) = result.conflicts
    assert conflict.path == ("a", "b")
    assert conflict.kind is ConflictKind.VALUE
    # The implicit super table a has no header source.
    assert conflict.base_span is not None
    assert conflict.base_span.text == 'a.b = 1\n'


def test_merge_documents_raises_on_conflicts() -> None:
    with pytest.raises(TOMLKitMergeError):
        merge_documents('a = 1\n', 'a = 2\n', 'a = 3\n')


def test_aot_edits_merge_with_identity_key() -> None:
    base = (
        '[[p]]\nid = 1\nv = 0\n'
        '[[p]]\nid = 2\nv = 0\n'
    )
    ours = (
        '[[p]]\nid = 1\nv = 1\n'
        '[[p]]\nid = 2\nv = 0\n'
    )
    theirs = (
        '[[p]]\nid = 1\nv = 0\n'
        '[[p]]\nid = 2\nv = 2\n'
    )

    result = merge3(base, ours, theirs, identity=identity_key("id"))

    assert result.clean
    assert _semantics(result) == {
        "p": [{"id": 1, "v": 1}, {"id": 2, "v": 2}]
    }


def test_aot_additions_on_both_sides_union_by_identity() -> None:
    base = '[[p]]\nid = 1\nv = 0\n'
    ours = (
        '[[p]]\nid = 1\nv = 0\n'
        '[[p]]\nid = 2\nv = 2\n'
    )
    theirs = (
        '[[p]]\nid = 3\nv = 3\n'
        '[[p]]\nid = 1\nv = 0\n'
    )

    result = merge3(base, ours, theirs, identity=identity_key("id"))

    assert result.clean
    assert _semantics(result) == {
        "p": [
            {"id": 1, "v": 0},
            {"id": 2, "v": 2},
            {"id": 3, "v": 3},
        ]
    }


def test_aot_reorder_with_identity_binds_edits_correctly() -> None:
    base = (
        '[[p]]\nid = 1\nv = 0\n'
        '[[p]]\nid = 2\nv = 0\n'
    )
    ours = (
        '[[p]]\nid = 2\nv = 0\n'
        '[[p]]\nid = 1\nv = 0\n'
    )
    theirs = (
        '[[p]]\nid = 1\nv = 9\n'
        '[[p]]\nid = 2\nv = 0\n'
    )

    result = merge3(base, ours, theirs, identity=identity_key("id"))

    assert result.clean
    assert _semantics(result)["p"] == [
        {"id": 2, "v": 0},
        {"id": 1, "v": 9},
    ]


def test_aot_without_identity_reports_candidates_on_reorder() -> None:
    base = '[[p]]\na = 1\n[[p]]\na = 2\n'
    ours = '[[p]]\na = 2\n[[p]]\na = 1\n'
    theirs = '[[p]]\na = 1\nx = 1\n[[p]]\na = 2\n'

    result = merge3(base, ours, theirs)

    (conflict,) = result.conflicts
    assert conflict.kind is ConflictKind.AOT_CANDIDATES
    assert conflict.path == ("p",)
    assert conflict.ours_value is not None


def test_aot_element_edit_without_identity_when_shape_unchanged() -> None:
    base = '[[p]]\na = 1\n[[p]]\na = 2\n'
    ours = '[[p]]\na = 10\n[[p]]\na = 2\n'
    theirs = '[[p]]\na = 1\n[[p]]\na = 20\n'

    result = merge3(base, ours, theirs)

    assert result.clean
    assert _semantics(result)["p"] == [{"a": 10}, {"a": 20}]


def test_aot_missing_identity_key_is_candidate_conflict() -> None:
    base = '[[p]]\nid = 1\n'
    ours = '[[p]]\nid = 1\nv = 1\n'
    theirs = '[[p]]\nv = 2\n'

    result = merge3(base, ours, theirs, identity=identity_key("id"))

    assert any(c.kind is ConflictKind.AOT_CANDIDATES for c in result.conflicts)


def test_aot_delete_element_vs_edit_conflicts_on_element_path() -> None:
    base = '[[p]]\nid = 1\nv = 0\n'
    result = merge3(
        base, '', '[[p]]\nid = 1\nv = 9\n', identity=identity_key("id")
    )

    (conflict,) = result.conflicts
    assert conflict.path == ("p", "@1")
    assert conflict.kind is ConflictKind.DELETE_DESCENDANTS

    resolved = result.resolve({("p", "@1"): Resolution.THEIRS})
    assert resolved.clean
    assert _semantics(resolved) == {"p": [{"id": 1, "v": 9}]}


def test_inline_table_expands_to_table_when_other_side_adds_header() -> None:
    base = '[s]\nt = {x = 1}\n'
    ours = '[s]\nt = {x = 1}\n'
    theirs = '[s.t]\nx = 1\ny = 2\n'

    result = merge3(base, ours, theirs)

    assert result.clean
    assert _semantics(result) == {"s": {"t": {"x": 1, "y": 2}}}
    # The result is valid structured TOML: t is a real [s.t] table.
    assert "[s.t]" in result.document.as_string()


def test_crlf_line_endings_survive_in_merged_output() -> None:
    base = 'a = 1\r\n[s]\r\nh = "x"\r\n'
    ours = 'a = 1\r\n[s]\r\nh = "x"\r\n'
    theirs = 'a = 2\r\n[s]\r\nh = "y"\r\n'

    result = merge3(base, ours, theirs)

    text = result.document.as_string()
    assert "\r\n" in text
    assert tomlkit.parse(text).unwrap() == {"a": 2, "s": {"h": "y"}}


def test_untouched_regions_keep_ours_bytes() -> None:
    ours = (
        "# header\n"
        'title = "mine"  # keep\n'
        "\n"
        "[server]\n"
        'host = "localhost"\n'
        "port = 8000 # port\n"
        "\n"
        "[server.tls]\n"
        "enabled = true\n"
        "\n"
        "[[users]]\n"
        'name = "a"\n'
        "age = 1\n"
        "\n"
        "[[users]]\n"
        'name = "b"\n'
        "age = 2\n"
    )
    base = ours.replace('"mine"', '"old"').replace(" # port", "")
    theirs = base.replace("enabled = true", "enabled = false").replace(
        'name = "a"\nage = 1', 'name = "a"\nage = 11'
    )

    result = merge3(base, ours, theirs)
    text = result.document.as_string()

    assert result.clean
    for untouched in (
        "# header\n",
        '  # keep\n',
        'host = "localhost"\n',
        "port = 8000 # port\n",
        '[[users]]\nname = "b"\nage = 2\n',
    ):
        assert untouched in text, untouched
    assert tomlkit.parse(text).unwrap()["server"]["tls"] == {"enabled": False}


def test_custom_item_resolution_recomputes_container_cleanly() -> None:
    result = merge3(
        '[t]\na = 1\nb = 2\n',
        '[t]\na = 10\nb = 20\n',
        '[t]\na = 100\nb = 20\n',
    )
    # ``a`` changed differently on both sides; ``b`` was changed the same way.
    (conflict,) = result.conflicts
    assert conflict.path == ("t", "a")
    resolved = result.resolve({("t", "a"): integer(100)})
    assert resolved.clean
    assert _semantics(resolved) == {"t": {"a": 100, "b": 20}}
    assert resolved.document.as_string().count("[t]") == 1


def test_conflict_resolution_keeps_sibling_trivia() -> None:
    result = merge3(
        'a = 1\nb = 2 # sibling\n', 'a = 10\nb = 2 # sibling\n', 'a = 20\nb = 2 # sibling\n'
    )
    (conflict,) = result.conflicts
    resolved = result.resolve({conflict.path: Resolution.OURS})
    text = resolved.document.as_string()
    assert text.count("# sibling") == 1
    assert "b = 2 # sibling\n" in text


def test_dotted_key_deletion_removes_whole_subtree() -> None:
    result = merge3(
        'a.b = 1\na.c = 2\nz = 0\n',
        'z = 0\n',
        'a.b = 1\na.c = 3\nz = 0\n',
    )
    (conflict,) = result.conflicts
    assert conflict.kind is ConflictKind.DELETE_DESCENDANTS
    resolved = result.resolve({conflict.path: Resolution.THEIRS})
    assert _semantics(resolved) == {"a": {"b": 1, "c": 3}, "z": 0}


def test_both_sides_added_different_keys_merges() -> None:
    result = merge3(
        "a = 1\n", "a = 1\nb = 2\n", "a = 1\nc = 3\n"
    )
    assert result.clean
    assert _semantics(result) == {"a": 1, "b": 2, "c": 3}


def test_added_same_key_with_same_value_merges() -> None:
    result = merge3("a = 1\n", "a = 1\nb = 2\n", "a = 1\nb = 2\n")
    assert result.clean
    assert _semantics(result) == {"a": 1, "b": 2}


def test_added_same_key_different_values_conflicts() -> None:
    result = merge3("a = 1\n", "a = 1\nb = 2\n", "a = 1\nb = 3\n")
    (conflict,) = result.conflicts
    assert conflict.kind is ConflictKind.VALUE
    assert conflict.base_value is None


def test_merge_accepts_documents() -> None:
    base = tomlkit.parse("a = 1\n")
    ours = tomlkit.parse("a = 2\n")
    theirs = tomlkit.parse("a = 1\n")
    result = merge3(base, ours, theirs)
    assert result.clean
    assert _semantics(result) == {"a": 2}


def test_aot_candidate_conflict_carries_spans() -> None:
    base = '[[p]]\na = 1\n[[p]]\na = 2\n'
    ours = '[[p]]\na = 2\n[[p]]\na = 1\n'
    theirs = '[[p]]\na = 1\nx = 1\n[[p]]\na = 2\n'

    result = merge3(base, ours, theirs)
    (conflict,) = result.conflicts

    assert conflict.kind is ConflictKind.AOT_CANDIDATES
    assert conflict.base_span is not None
    assert conflict.ours_span is not None
    assert conflict.theirs_span is not None
    assert conflict.base_span.text.startswith("[[p]]")


def test_dotted_key_leaf_span_covers_full_entry() -> None:
    result = merge3('a.b.c = 1\n', 'a.b.c = 2\n', 'a.b.c = 3\n')
    (conflict,) = result.conflicts
    assert conflict.path == ("a", "b", "c")
    assert conflict.base_span is not None
    assert conflict.base_span.text == 'a.b.c = 1\n'
    assert conflict.base_span.start_line == 1
    assert conflict.base_span.start_column == 1


def test_multiline_inline_table_expansion_is_valid_toml() -> None:
    base = 't = {\n  a = 1, # first\n  b = 2,\n}\n'
    ours = base
    theirs = '[t]\na = 1\nc = 3\n'

    result = merge3(base, ours, theirs)

    assert result.clean
    semantics = tomlkit.parse(result.document.as_string()).unwrap()
    assert semantics == {"t": {"a": 1, "c": 3}}


def test_independent_aot_additions_with_no_base() -> None:
    result = merge3(
        '',
        '[[p]]\nid = 1\n',
        '[[p]]\nid = 2\n',
        identity=identity_key("id"),
    )
    assert result.clean
    assert _semantics(result) == {"p": [{"id": 1}, {"id": 2}]}


def test_conflict_preview_is_valid_and_keeps_ours() -> None:
    result = merge3('a = 1\nb = 2\n', 'a = 10\nb = 2\n', 'a = 20\nb = 3\n')

    # Preview document must always parse and reflect ours for the conflict.
    preview = tomlkit.parse(result.document.as_string()).unwrap()
    assert preview == {"a": 10, "b": 3}

    resolved = result.resolve({"a": Resolution.THEIRS})
    assert resolved.clean
    assert _semantics(resolved) == {"a": 20, "b": 3}


def test_original_documents_are_not_mutated_by_merge() -> None:
    base = tomlkit.parse('a = 1\n')
    ours = tomlkit.parse('a = 2\n')
    theirs = tomlkit.parse('a = 3\n')

    merge3(base, ours, theirs)

    assert base.as_string() == 'a = 1\n'
    assert ours.as_string() == 'a = 2\n'
    assert theirs.as_string() == 'a = 3\n'


def test_resolve_with_dotted_string_path() -> None:
    result = merge3('[t]\na = 1\n', '[t]\na = 2\n', '[t]\na = 3\n')

    resolved = result.resolve({"t.a": Resolution.OURS})

    assert resolved.clean
    assert _semantics(resolved) == {"t": {"a": 2}}


def test_divergent_table_header_comments_conflict() -> None:
    base = '[s] # base\nx = 1\n'
    ours = '[s] # ours\nx = 1\n'
    theirs = '[s] # theirs\nx = 2\n'

    result = merge3(base, ours, theirs)

    (conflict,) = result.conflicts
    assert conflict.kind is ConflictKind.COMMENT
    assert conflict.path == ("s",)

    resolved = result.resolve({("s",): Resolution.THEIRS})
    assert resolved.clean
    assert "[s] # theirs\n" in resolved.document.as_string()


def test_table_header_comment_only_edit_merges_with_value_edit() -> None:
    result = merge3('[s]\nx = 1\n', '[s] # note\nx = 1\n', '[s]\nx = 2\n')

    assert result.clean
    text = result.document.as_string()
    assert "[s] # note\n" in text
    assert "x = 2\n" in text


def test_both_sides_promote_scalar_to_same_table_merges() -> None:
    result = merge3('k = 1\n', '[k]\nx = 1\n', '[k]\nx = 1\n')
    assert result.clean
    assert _semantics(result) == {"k": {"x": 1}}


def test_both_sides_promote_scalar_to_table_with_different_children_merges() -> None:
    result = merge3('k = 1\n', '[k]\nx = 1\n', '[k]\ny = 2\n')
    assert result.clean
    assert _semantics(result) == {"k": {"x": 1, "y": 2}}


def test_both_sides_promote_scalar_to_inline_table_merges() -> None:
    result = merge3('k = 1\n', 'k = {x = 1}\n', 'k = {x = 1}\n')
    assert result.clean
    assert _semantics(result) == {"k": {"x": 1}}


def test_both_sides_promote_scalar_to_same_aot_merges() -> None:
    result = merge3('k = 1\n', '[[k]]\nx = 1\n', '[[k]]\nx = 1\n')
    assert result.clean
    assert _semantics(result) == {"k": [{"x": 1}]}


def test_both_sides_promote_scalar_to_divergent_aot_is_candidate() -> None:
    result = merge3('k = 1\n', '[[k]]\nx = 1\n', '[[k]]\nx = 2\n')
    assert [c.kind for c in result.conflicts] == [ConflictKind.AOT_CANDIDATES]


def test_spans_report_lines_and_columns() -> None:
    from tomlkit.merge import parse_with_spans

    source = 'a = 1\n[foo]\nx = 1\n'
    doc, spans = parse_with_spans(source)
    span = spans[id(doc['foo'].item('x'))]

    assert span.start_line == 3
    assert span.start_column == 1
    assert span.end_line == 3
    assert span.text == 'x = 1\n'
    assert span.as_dict()["text"] == 'x = 1\n'
