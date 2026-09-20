Structured Three-Way Merge
==========================

A line-oriented three-way merge treats TOML as plain text. That is fine for
independent lines, but TOML has structure that text tools do not understand:

* a **dotted key** (``a.b.c = 1``) and an **explicit table** (``[a.b]``) can
  define the *same* semantic path;
* a **table can be split across several headers** (including out-of-order
  ``[a]`` fragments), so moving whole blocks can duplicate a header or produce
  an out-of-order table;
* an **array of tables** (``[[servers]]``) is an ordered collection whose
  elements have no inherent position-based identity.

A textual merge can therefore emit a document that parses but means the wrong
thing. :func:`tomlkit.merge` merges the *semantic* TOML tree instead: it aligns
nodes by semantic path and container type, rebuilds the affected containers
through tomlkit's own mutation API, and leaves every region it does not touch as
the exact bytes taken from *ours*.

Basic use
---------

.. code-block:: python

    from tomlkit import merge

    result = merge(base_text, ours_text, theirs_text)

    if result.ok:
        merged = result.merged  # a TOMLDocument
    else:
        for conflict in result.conflicts:
            ...

The inputs may be source strings or documents parsed with
:func:`tomlkit.parse_for_merge`. Passing strings is equivalent; the span
evidence described below is computed internally either way.

The result always contains a structurally valid ``merged`` document even when
conflicts remain: unresolvable nodes keep *ours*, so the document is always
safe to inspect or write to disk.

Identity and what merges automatically
--------------------------------------

Nodes are identified by their **semantic path** (the tuple of keys from the
root) together with their **container type**:

* Edits to *different keys* in the same table always merge.
* When both sides edit the same scalar, equal resulting values merge;
  different resulting values raise a :class:`tomlkit.Conflict` of kind
  ``VALUE``.
* Deleting a container while the other side modifies a value inside it raises
  ``DELETE_MODIFIED``; modifying a deeper descendant raises
  ``DELETE_DESCENDANT``.
* A scalar-to-scalar type change (for example date versus string, integer
  versus float) is a ``VALUE`` conflict. A change between container shapes
  (table, inline table, array of tables, scalar) is a ``STRUCTURE`` conflict.

Comment policy
--------------

A comment is metadata on the item it decorates, independent of its value. If
one side changes a value and the other side changes only that item's comment,
the combination is ambiguous for tools, so the default
:class:`tomlkit.CommentPolicy.CONFLICT` reports a ``COMMENT`` conflict. Pass
``comment_policy=CommentPolicy.MERGE`` to take the new value from the
value-changing side and the new comment from the comment-changing side.

Arrays of tables and identity keys
-----------------------------------

Elements of an array of tables **must not be matched by position**: inserting
or removing an element shifts every later index, and two different arrays can
happen to line up element-for-element while describing unrelated records.

Provide an *identity* mapping so the merger can pair elements by a stable
key. The mapping key is the AoT path (a tuple of key strings, or a dotted
string); the value is either the name of the field that uniquely identifies an
element, or a callable receiving the unwrapped element dict and returning a
hashable identity:

.. code-block:: python

    result = merge(
        base, ours, theirs,
        identity={
            ("servers",): "name",
            "services.replicas": lambda element: (element["region"], element["id"]),
        },
    )

With an identity key, edits to different fields of matched elements,
independent inserts/deletes, and single-sided reorders merge automatically.
Both sides reordering the shared elements *differently* raises an
``AOT_REORDER`` conflict, and deleting an element that the other side modifies
raises ``DELETE_MODIFIED`` / ``DELETE_DESCENDANT`` just like regular tables.

Why some arrays cannot be merged automatically
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Without an identity key there is no principled way to tell whether an element
in *ours* is "the same record" as the element at the same index in *theirs*.
The merger therefore **never guesses by position**:

* if only one side (or neither) changed the array, that side is taken;
* if both sides modified an identity-less array, the whole array raises
  ``AOT_AMBIGUOUS``, and the conflict carries conservative ``candidates`` (the
  semantic paths of the records on each side) for the caller to inspect.

Duplicate identity values, or two sides adding the same identity with
different content, are likewise reported instead of resolved arbitrarily.

Conflicts and source evidence
-----------------------------

Each :class:`tomlkit.Conflict` exposes:

* ``path`` -- the semantic path of the node;
* ``kind`` -- the :class:`tomlkit.ConflictKind` reason;
* ``base`` / ``ours`` / ``theirs`` -- the three tomlkit items (or ``None`` for
  missing sides);
* ``base_span`` / ``ours_span`` / ``theirs_span`` --
  :class:`tomlkit.SourceRange` objects with ``start``/``end`` offsets and the
  exact ``text`` each side occupies;
* ``aot_index`` and ``candidates`` for array-of-tables ambiguities.

Resolving conflicts
-------------------

Call :meth:`tomlkit.MergeResult.apply` to build the final document. Provide a
mapping of path to resolution, or a callable that receives each conflict and
returns one of:

* ``"ours"`` (the conservative default) -- keep our version;
* ``"theirs"`` -- take the incoming version;
* ``"delete"`` -- remove the node;
* an :class:`tomlkit.items.Item` (or a plain Python value encoded via
  :func:`tomlkit.item`) used as the replacement.

After resolving, the affected containers are recomputed structurally, so the
output cannot contain duplicate tables, out-of-order headers, or swallowed
sibling trivia.

Line endings
------------

Untouched lines keep *ours*' exact representation, including CRLF endings and
surrounding comments; only containers actually involved in the merge are
re-rendered.
