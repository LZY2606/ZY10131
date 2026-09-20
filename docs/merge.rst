Structured three-way merge
==========================

.. versionadded:: 0.16.0

A textual three-way merge treats a TOML document as a bag of lines. That works
for unrelated one-line edits, but TOML structure spans multiple lines and has
several spellings for the same semantic content:

* a dotted key (``a.b = 1``) and an explicit header (``[a]`` then ``b = 1``)
  describe the *same* table;
* ``[[servers]]`` introduces an **array of tables** whose elements can be
  reordered, split across headers, or interleaved with other tables;
* an inline table (``t = { x = 1 }``) is semantically a table.

Line-oriented merging can therefore return text that parses but means
something different (edits applied to the wrong array element, a table
duplicated, a header moved out of order). ``tomlkit.merge`` merges the
*semantic* trees instead, then renders a document: regions neither side
touched keep their exact original representation (comments, whitespace,
quoting and CRLF endings included).

Basic usage
-----------

.. code-block:: python

    from tomlkit import merge3, Resolution

    result = merge3(base_text, ours_text, theirs_text)
    if result.clean:
        merged_text = result.document.as_string()
    else:
        for conflict in result.conflicts:
            print(conflict.path_string, conflict.kind.value,
                  conflict.base_span and conflict.base_span.text)
        # Pick a side per conflict, or supply a custom tomlkit item.
        result = result.resolve({
            ("server", "port"): Resolution.THEIRS,
            ("server", "host"): "example.com",
        })
        merged_text = result.merged_document().as_string()

``merge3`` returns a :class:`~tomlkit.merge.MergeResult`. Its ``document`` is
always valid TOML — while conflicts remain it is a *preview* keeping the
``ours`` state for the unresolved paths. ``resolve`` recomputes the whole
merge with your decisions and returns a new result; no partial mutation is
ever left behind.

Identity
--------

Merge identity is the **semantic path plus the container type**:

* ``a.b`` written as a dotted key is the same node as ``b`` under ``[a]``;
* a plain table, an inline table and an array of tables at one path are
  distinct container types;
* edits to *different* keys merge automatically;
* deleting a container while the other side edits descendants is always a
  :attr:`~tomlkit.merge.ConflictKind.DELETE_DESCENDANTS` conflict;
* changing the same scalar value on both sides is a
  :attr:`~tomlkit.merge.ConflictKind.VALUE` conflict, while changing the
  *shape* (table vs. inline table vs. array of tables vs. scalar) is a
  :attr:`~tomlkit.merge.ConflictKind.STRUCTURE` conflict. An inline table
  expanded into a regular table on one side merges with the inline spelling
  of the other side.

Comments
~~~~~~~~

If one side changes an entry value and the other side changes only that
entry's comment, the default policy
(``comment_policy=CommentConflictPolicy.MERGE``) keeps both changes. Pass
``CommentConflictPolicy.CONFLICT`` to surface such a collision as a
:attr:`~tomlkit.merge.ConflictKind.COMMENT` conflict instead. Comments edited
differently on both sides always conflict.

Arrays of tables
----------------

Plain arrays (``a = [1, 2]``) are values: the whole array is compared as one
node. An **array of tables** is a sequence of containers, so it needs a way to
know which element on one side corresponds to which element on the other.

Element position is *not* an identity — a reorder would otherwise move every
edit to the wrong table. Instead, pass an identity policy:

.. code-block:: python

    from tomlkit import merge3, identity_key

    result = merge3(base, ours, theirs, identity=identity_key("name"))
    # nested fields work too: identity_key("meta", "id") or "meta.id"

Elements are then matched by the stable field: independent edits merge,
additions union, deletions combine with edits per element (an element deleted
by one side and edited by the other is a
:attr:`~tomlkit.merge.ConflictKind.DELETE_DESCENDANTS` conflict at
``aot.@<identity>``), and reorders are respected.

Why some arrays of tables cannot merge automatically
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Without a usable identity policy there is no sound way to tell whether two
divergences mean "element 0 was edited" or "element 0 was removed and a new
element was inserted before it". In that case the merge stays conservative:

* when the array length is unchanged on both sides and at most one side
  differs from base, corresponding elements are merged by position;
* any simultaneous add/remove/reorder divergence — or an element whose
  identity key is missing/duplicated — is reported as a single
  :attr:`~tomlkit.merge.ConflictKind.AOT_CANDIDATES` conflict with all three
  sides' source spans, instead of guessing a matching that could silently
  rebind edits.

Conflict reference
------------------

Each :class:`~tomlkit.merge.Conflict` exposes:

* ``path`` / ``path_string`` — the semantic path (``@<id>`` marks an AoT
  element);
* ``base_value``, ``ours_value``, ``theirs_value`` — the raw tomlkit items or
  ``None`` when the path is absent (added or deleted);
* ``base_span``, ``ours_span``, ``theirs_span`` — :class:`SourceSpan`
  evidence with offsets, 1-based line/columns and the exact source text;
* ``kind`` — one of :class:`~tomlkit.merge.ConflictKind`;
* ``message`` — a human-readable explanation.

Resolve a conflict with :class:`~tomlkit.merge.Resolution.OURS`,
:class:`~tomlkit.merge.Resolution.THEIRS`, or any tomlkit item (for example
``tomlkit.integer(8080)``). Resolutions may be supplied repeatedly; the merge
is recomputed, so replacing or removing a container never leaves a duplicate
``[table]`` header, an out-of-order table, or clobbers sibling trivia.

API
---

.. automodule:: tomlkit.merge
   :members:
   :undoc-members:
   :show-inheritance:
