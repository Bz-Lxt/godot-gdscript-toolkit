"""Range (partial) formatting.

Formats only the smallest complete syntactic unit covering the line range
requested by the caller (e.g. an editor selection). Every byte outside the
expanded range is preserved verbatim.

The strategy is:
1. parse the whole file (with positions) and expand the requested range to
   the minimal complete syntactic unit (one statement or a contiguous group
   of sibling statements, incl. attached annotations);
2. format the whole file as usual - this guarantees the formatted fragment is
   exactly what full-file formatting would produce, so formatting a range of
   already formatted code is a no-op (idempotency);
3. splice the formatted fragment back into the *original* code, replacing
   whole physical lines only, so content before/after the fragment is kept
   byte-for-byte.
"""
import re
from dataclasses import dataclass
from math import gcd
from typing import List, Optional

from lark import Tree, Token

from ..parser import parser

# Nodes that are complete statements at class or function scope.
STATEMENT_RULES = frozenset(
    [
        # class scope
        "class_def",
        "property_body_def",
        "func_def",
        "static_func_def",
        "abstract_func_def",
        "pass_stmt",
        "enum_stmt",
        "signal_stmt",
        "extends_stmt",
        "classname_extends_stmt",
        "classname_stmt",
        "class_var_stmt",
        "static_class_var_stmt",
        "const_stmt",
        "docstr_stmt",
        "annotation",
        # function scope
        "return_stmt",
        "func_var_stmt",
        "break_stmt",
        "breakpoint_stmt",
        "continue_stmt",
        "expr_stmt",
        "if_stmt",
        "while_stmt",
        "for_stmt",
        "for_stmt_typed",
        "match_stmt",
    ]
)

# Nodes whose direct children form a scope we may descend into.
SCOPE_RULES = frozenset(
    [
        "start",
        "class_def",
        "func_def",
        "static_func_def",
        "if_branch",
        "elif_branch",
        "else_branch",
        "match_branch",
        "guarded_match_branch",
        "property_body_def",
        "property_custom_getter",
        "property_custom_setter",
        "while_stmt",
        "for_stmt",
        "for_stmt_typed",
    ]
)

# Compound statements which can themselves be returned as the unit when only
# their header line is selected.
COMPOUND_SCOPE_RULES = SCOPE_RULES | frozenset(["if_stmt", "match_stmt"])

# Branch wrapper nodes (children of if_stmt / match_stmt).
BRANCH_RULES = frozenset(
    ["if_branch", "elif_branch", "else_branch", "match_branch", "guarded_match_branch"]
)

# Nodes whose ``meta`` positions are skewed by synthetic INDENT/DEDENT/NL
# tokens emitted by the indenter. For these we compute spans recursively from
# their children.
BLOCK_BEARING_RULES = frozenset(
    [
        "start",
        "class_def",
        "func_def",
        "static_func_def",
        "if_stmt",
        "if_branch",
        "elif_branch",
        "else_branch",
        "while_stmt",
        "for_stmt",
        "for_stmt_typed",
        "match_stmt",
        "match_branch",
        "guarded_match_branch",
        "property_body_def",
        "property_custom_getter",
        "property_custom_setter",
        "lambda",
    ]
)

# Nodes used when mapping original fragments to formatted fragments.
CORRESPONDENCE_RULES = STATEMENT_RULES | frozenset(
    [
        "match_branch",
        "guarded_match_branch",
        "property_custom_getter",
        "property_custom_setter",
    ]
)

_SYNTHETIC_TOKEN_TYPES = frozenset(["_NL", "_INDENT", "_DEDENT"])


@dataclass
class Range:
    """1-based, inclusive line range."""

    start: int
    end: int


@dataclass
class RangeFormattingOutcome:
    code: str
    original_range: Range
    """minimal unit expanded from the requested range, in the source file"""
    formatted_range: Optional[Range]
    """location of the formatted fragment in the resulting code
    (``None`` when the selection was comments/blank lines only)"""


def node_start_line(node) -> int:
    if isinstance(node, Token):
        return node.line
    if node.data in BLOCK_BEARING_RULES:
        starts = [
            node_start_line(child)
            for child in node.children
            if _has_real_position(child)
        ]
        return min(starts)
    return node.meta.line


def node_end_line(node) -> int:
    if isinstance(node, Token):
        return node.end_line if node.end_line is not None else node.line
    if node.data in BLOCK_BEARING_RULES:
        ends = [
            node_end_line(child)
            for child in node.children
            if _has_real_position(child)
        ]
        return max(ends)
    return node.meta.end_line


def _has_real_position(node) -> bool:
    if isinstance(node, Token):
        return node.type not in _SYNTHETIC_TOKEN_TYPES
    return True


def _covers(node, selected: Range) -> bool:
    return node_start_line(node) <= selected.start and node_end_line(node) >= selected.end


def _intersects(node, selected: Range) -> bool:
    return node_start_line(node) <= selected.end and node_end_line(node) >= selected.start


def _starts_at_line_beginning(node, code_lines: List[str]) -> bool:
    start = node_start_line(node)
    if start < 1 or start > len(code_lines):
        return True
    prefix = code_lines[start - 1][: node.meta.column - 1]
    return prefix.strip() == ""


def resolve_unit(
    parse_tree: Tree, code_lines: List[str], selected: Range
) -> Optional[List[Tree]]:
    """Expands the selected range to the minimal complete syntactic unit.

    Returns the parse tree nodes the unit consists of (one statement, or a
    contiguous group of sibling statements incl. leading annotations), or
    ``None`` when the selection covers comments/blank lines only.
    """
    parent = {}
    for subtree in parse_tree.iter_subtrees():
        for child in subtree.children:
            if isinstance(child, Tree):
                parent[id(child)] = subtree
    nodes = _resolve_in_scope(parse_tree, selected, code_lines)
    if nodes is None:
        return None
    first = nodes[0]
    if _starts_at_line_beginning(first, code_lines):
        return nodes
    # e.g. the "extends X" part of "class Foo extends X:" - the node starts
    # in the middle of a line, so climb to the statement enclosing it.
    ancestor = parent.get(id(first))
    while ancestor is not None:
        if ancestor.data in STATEMENT_RULES and _starts_at_line_beginning(
            ancestor, code_lines
        ):
            return [ancestor]
        ancestor = parent.get(id(ancestor))
    return nodes


def _member_children(scope: Tree) -> List[Tree]:
    member_rules = STATEMENT_RULES | (SCOPE_RULES - BRANCH_RULES)
    return [
        child
        for child in scope.children
        if isinstance(child, Tree) and child.data in member_rules
    ]


def _selection_starts_in_body(compound: Tree, selected: Range) -> bool:
    """True iff the first selected line is within a body statement (i.e. the
    selection does not start on the compound's header)."""
    if compound.data in ("if_stmt", "match_stmt"):
        for branch in compound.children:
            if not (isinstance(branch, Tree) and branch.data in BRANCH_RULES):
                continue
            if not _intersects(branch, selected):
                continue
            body_hits = [
                m for m in _member_children(branch) if _intersects(m, selected)
            ]
            # do not descend when header and body share the selected physical
            # line (e.g. "if x: y()" or "get: pass" on one line)
            return (
                bool(body_hits)
                and node_start_line(body_hits[0]) <= selected.start
                and node_start_line(branch) < selected.start
            )
        return False
    body_hits = [m for m in _member_children(compound) if _intersects(m, selected)]
    return (
        bool(body_hits)
        and node_start_line(body_hits[0]) <= selected.start
        and node_start_line(compound) < selected.start
    )


def _selection_is_comment_or_blank(
    selected: Range, code_lines: List[str]
) -> bool:
    for line_no in range(selected.start, selected.end + 1):
        if not 1 <= line_no <= len(code_lines):
            continue
        stripped = code_lines[line_no - 1].strip()
        if stripped != "" and not stripped.startswith("#"):
            return False
    return True


def _selection_is_inner_comment_only(
    compound: Tree, selected: Range, code_lines: List[str]
) -> bool:
    return (
        selected.start > node_start_line(compound)
        and selected.end <= node_end_line(compound)
        and _selection_is_comment_or_blank(selected, code_lines)
    )


# pylint: disable-next=too-many-return-statements
def _resolve_in_scope(
    scope: Tree, selected: Range, code_lines: List[str]
) -> Optional[List[Tree]]:
    if scope.data in ("if_stmt", "match_stmt"):
        branches = [
            child
            for child in scope.children
            if isinstance(child, Tree) and child.data in BRANCH_RULES
        ]
        intersecting_branches = [b for b in branches if _intersects(b, selected)]
        if len(intersecting_branches) == 1:
            if _selection_starts_in_body(scope, selected):
                resolved = _resolve_in_scope(
                    intersecting_branches[0], selected, code_lines
                )
                if resolved is not None:
                    return resolved
            elif _selection_is_inner_comment_only(scope, selected, code_lines):
                # comment/blank lines inside a branch body: nothing to format
                return None
        # header line, or selection spanning several branches -> whole statement
        return [scope]

    member_children = _member_children(scope)
    hits = [child for child in member_children if _intersects(child, selected)]
    if not hits:
        if scope.data in (
            STATEMENT_RULES
            | frozenset(["property_custom_getter", "property_custom_setter"])
        ) and not _selection_is_comment_or_blank(selected, code_lines):
            # header-only line (or multiline header continuation) of a
            # compound statement
            return [scope]
        # comment/blank-line-only selection -> nothing to format
        return None
    if len(hits) == 1:
        hit = hits[0]
        if hit.data in COMPOUND_SCOPE_RULES:
            if _selection_starts_in_body(hit, selected):
                resolved = _resolve_in_scope(hit, selected, code_lines)
                if resolved is not None:
                    return resolved
            elif _selection_is_inner_comment_only(hit, selected, code_lines):
                # comment/blank lines inside the compound body
                return None
        return [hit]

    first_index = _identity_index(member_children, hits[0])
    last_index = _identity_index(member_children, hits[-1])
    group = member_children[first_index : last_index + 1]
    return _prepend_leading_annotations(scope, member_children[first_index], group)


def _identity_index(nodes: List[Tree], target: Tree) -> int:
    """Identity-based ``list.index`` - Lark trees compare structurally, so
    repeated identical nodes (e.g. several ``@abstract`` annotations) are
    indistinguishable with the equality-based lookup."""
    return next(i for i, node in enumerate(nodes) if node is target)


def _prepend_leading_annotations(
    scope: Tree, first_member: Tree, group: List[Tree]
) -> List[Tree]:
    raw_children = scope.children
    try:
        index = _identity_index(raw_children, first_member)
    except StopIteration:
        return group
    while index - 1 >= 0:
        previous = raw_children[index - 1]
        if isinstance(previous, Tree) and previous.data == "annotation":
            group = [previous] + group
            index -= 1
            continue
        break
    return group


def _enumerate_units(parse_tree: Tree) -> List[Tree]:
    return [
        node
        for node in parse_tree.iter_subtrees()
        if node.data in CORRESPONDENCE_RULES
    ]


def _build_parent_map(parse_tree: Tree) -> dict:
    parent_map = {}
    for subtree in parse_tree.iter_subtrees():
        for child in subtree.children:
            if isinstance(child, Tree):
                parent_map[id(child)] = subtree
    return parent_map


def _is_related(node: Tree, other_id: int, parent_map: dict) -> bool:
    """True iff ``node`` is an ancestor of the node identified by
    ``other_id`` (or vice versa - handled by the caller's checks)."""
    ancestor = parent_map.get(other_id)
    while ancestor is not None:
        if ancestor is node:
            return True
        ancestor = parent_map.get(id(ancestor))
    return False


def _close_over_shared_lines(
    seed_nodes: List[Tree], all_units: List[Tree], parent_map: dict
) -> List[Tree]:
    """Expands the seed to every unit rendered on one of the same physical
    lines, ignoring enclosing/dominated nodes (an ancestor block always
    overlaps its descendants but must not grow the region)."""
    nodes = list(seed_nodes)
    node_ids = {id(node) for node in nodes}
    start = min(node_start_line(node) for node in nodes)
    end = max(node_end_line(node) for node in nodes)
    while True:
        added = []
        for node in all_units:
            if id(node) in node_ids:
                continue
            if node_start_line(node) > end or node_end_line(node) < start:
                continue
            if any(
                _is_related(node, other_id, parent_map) or other_id == id(node)
                for other_id in node_ids
            ):
                continue
            added.append(node)
        if not added:
            return nodes
        nodes.extend(added)
        node_ids.update(id(node) for node in added)
        start = min(start, *(node_start_line(node) for node in added))
        end = max(end, *(node_end_line(node) for node in added))


# pylint: disable-next=too-many-locals,too-many-branches
def _align_units(
    original_units: List[Tree], formatted_units: List[Tree]
) -> dict:
    """Aligns original units with formatted units in source order.

    Returns a mapping ``id(original node) -> [formatted nodes]``. Formatting
    can both merge several original units into a single rendered unit (e.g.
    ``@export`` annotations joined onto the following statement line) and
    expand one original unit into several formatted nodes (one-line property
    delegates becoming ``property_body_def`` blocks). Mapped nodes on either
    side thus form equivalence classes; callers expand the selected region to
    the whole class so that fragments line up with their surroundings.
    """
    mapping = {id(original): [] for original in original_units}
    i = 0
    j = 0
    previous_original = None
    total_formatted = len(formatted_units)
    total_original = len(original_units)

    def attach_expanded_property_body():
        nonlocal j, previous_original
        if previous_original is None:
            raise ValueError(
                "Could not map the syntactic unit onto formatted code"
            )
        mapping[id(previous_original)].append(formatted_units[j])
        j += 1

    while i < total_original:
        original = original_units[i]
        while (
            j < total_formatted
            and formatted_units[j].data == "property_body_def"
            and original.data != "property_body_def"
        ):
            attach_expanded_property_body()

        if original.data == "annotation":
            annotation_group = []
            while (
                i < total_original and original_units[i].data == "annotation"
            ):
                annotation_group.append(original_units[i])
                i += 1
            if i >= total_original:
                # dangling annotation(s) at the end of the file
                for annotation in annotation_group:
                    if j < total_formatted:
                        mapping[id(annotation)].append(formatted_units[j])
                        j += 1
                break
            statement = original_units[i]
            group = annotation_group + [statement]
            if j < total_formatted and formatted_units[j].data == "annotation":
                # annotations kept on their own lines - pair 1:1, the rest
                # gets merged with the statement line
                remaining = list(group)
                while (
                    remaining
                    and j < total_formatted
                    and formatted_units[j].data == "annotation"
                    and remaining[0].data == "annotation"
                ):
                    mapping[id(remaining.pop(0))].append(formatted_units[j])
                    j += 1
                if j >= total_formatted or formatted_units[j].data != statement.data:
                    raise ValueError(
                        "Could not map the syntactic unit onto formatted code"
                    )
                rendered = formatted_units[j]
                j += 1
                for node in remaining:
                    mapping[id(node)].append(rendered)
            else:
                if j >= total_formatted or formatted_units[j].data != statement.data:
                    raise ValueError(
                        "Could not map the syntactic unit onto formatted code"
                    )
                rendered = formatted_units[j]
                j += 1
                for node in group:
                    mapping[id(node)].append(rendered)
            previous_original = statement
            i += 1
            continue

        if (
            j >= total_formatted
            or formatted_units[j].data != original.data
        ):
            raise ValueError("Could not map the syntactic unit onto formatted code")
        mapping[id(original)].append(formatted_units[j])
        previous_original = original
        i += 1
        j += 1

    while j < total_formatted and formatted_units[j].data == "property_body_def":
        attach_expanded_property_body()
    return mapping


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def format_code_in_range(
    gdscript_code: str,
    range_start: int,
    range_end: int,
    max_line_length: int,
    spaces_for_indent: Optional[int] = None,
    parse_tree: Optional[Tree] = None,
    comment_parse_tree: Optional[Tree] = None,
) -> RangeFormattingOutcome:
    """Formats the minimal unit covering ``range_start``-``range_end``.

    Raises ``ValueError`` for invalid ranges.
    """
    if range_start < 1 or range_end < range_start:
        raise ValueError("Invalid range: expected 1 <= range_start <= range_end")

    parse_tree = (
        parse_tree
        if parse_tree is not None
        else parser.parse(gdscript_code, gather_metadata=True)
    )
    code_lines = gdscript_code.splitlines()
    if range_end > len(code_lines):
        raise ValueError("Range reaches beyond the end of the file")
    selected = Range(range_start, range_end)
    unit_nodes = resolve_unit(parse_tree, code_lines, selected)
    if unit_nodes is None:
        return RangeFormattingOutcome(gdscript_code, Range(range_start, range_end), None)

    # imported lazily to avoid an import cycle (formatter imports this module)
    from .formatter import (  # pylint: disable=import-outside-toplevel
        format_code as _format_full_code,
    )

    formatted_code = _format_full_code(
        gdscript_code,
        max_line_length,
        spaces_for_indent=spaces_for_indent,
        parse_tree=parse_tree,
        comment_parse_tree=comment_parse_tree,
    )
    formatted_tree = parser.parse(formatted_code, gather_metadata=True)

    original_units = _enumerate_units(parse_tree)
    formatted_units = _enumerate_units(formatted_tree)
    mapping = _align_units(original_units, formatted_units)

    # Splicing replaces whole physical lines, so on the original side group
    # all units sharing a line with the selection (e.g. ';'-separated
    # statements)...
    original_parent_map = _build_parent_map(parse_tree)
    original_nodes = _close_over_shared_lines(
        unit_nodes, original_units, original_parent_map
    )
    # ...and on the formatted side pull in units rendered onto the same line
    # (e.g. an annotation joined with its statement line).
    formatted_nodes = [
        node for unit in original_nodes for node in mapping[id(unit)]
    ]
    formatted_parent_map = _build_parent_map(formatted_tree)
    formatted_nodes = _close_over_shared_lines(
        formatted_nodes, formatted_units, formatted_parent_map
    )
    # finally keep the original region in sync with every original unit whose
    # rendered counterpart participates in the formatted fragment
    included_formatted_ids = {id(node) for node in formatted_nodes}
    expanded_original_nodes = [
        node
        for node in original_units
        if any(id(formatted) in included_formatted_ids for formatted in mapping[id(node)])
    ]
    if not expanded_original_nodes or not formatted_nodes:
        raise ValueError("Could not map the syntactic unit onto formatted code")
    original_start = min(node_start_line(node) for node in expanded_original_nodes)
    original_end = min(
        max(node_end_line(node) for node in expanded_original_nodes),
        len(code_lines),
    )
    formatted_start = min(node_start_line(node) for node in formatted_nodes)
    formatted_end = min(
        max(node_end_line(node) for node in formatted_nodes),
        len(formatted_code.splitlines()),
    )

    configured_unit = "\t" if spaces_for_indent is None else " " * spaces_for_indent
    result = _splice_lines(
        gdscript_code,
        original_start,
        original_end,
        formatted_code,
        formatted_start,
        formatted_end,
        configured_unit,
    )
    formatted_range = Range(
        original_start, original_start + (formatted_end - formatted_start)
    )
    return RangeFormattingOutcome(
        result, Range(original_start, original_end), formatted_range
    )


def _detect_indent_unit(code_lines: List[str]) -> str:
    """Infers the indentation unit (a tab or N spaces) used by the file."""
    space_widths = []
    for line in code_lines:
        match = re.match(r"^\t+", line)
        if match is not None:
            return "\t"
        match = re.match(r"^ +", line)
        if match is not None:
            space_widths.append(len(match.group(0)))
    if not space_widths:
        return "\t"
    width = space_widths[0]
    for value in space_widths[1:]:
        width = gcd(width, value)
    return " " * max(width, 1)


def _indent_levels(indent: str, unit: str) -> int:
    if unit == "\t":
        return indent.count("\t")
    return len(indent) // len(unit)


def _reindent_fragment(
    fragment_lines: List[str],
    configured_unit: str,
    source_unit: str,
    source_units_per_level: float,
) -> List[str]:
    """Rewrites leading indentation of the formatted fragment so that it uses
    the source file's indentation unit and indentation width per level (the
    formatter always emits one tab per level, while a file may use e.g. two
    tabs or four spaces). Lines inside multiline strings are left untouched."""
    result = []
    inside_multiline_string = False
    for line in fragment_lines:
        quote_count = len(re.findall('"""|\'\'\'', line))
        if inside_multiline_string:
            result.append(line)
            if quote_count % 2 == 1:
                inside_multiline_string = False
            continue
        if quote_count % 2 == 1:
            inside_multiline_string = True
        match = re.match(r"^[\t ]*", line)
        indent = match.group(0)
        if configured_unit == "\t":
            levels = indent.count("\t")
            remainder = indent[levels:]
        else:
            levels, remainder_units = divmod(len(indent), len(configured_unit))
            remainder = " " * remainder_units
        source_levels = int(round(levels * source_units_per_level))
        result.append(
            source_unit * source_levels + remainder + line[len(indent) :]
        )
    return result


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def _splice_lines(
    original_code: str,
    original_start: int,
    original_end: int,
    formatted_code: str,
    formatted_start: int,
    formatted_end: int,
    configured_unit: str,
) -> str:
    original_lines = original_code.splitlines(keepends=True)
    formatted_lines = formatted_code.splitlines(keepends=True)

    before = original_lines[: original_start - 1]
    after = original_lines[original_end:]
    fragment_lines = formatted_lines[formatted_start - 1 : formatted_end]
    source_unit = _detect_indent_unit(original_code.splitlines())

    # The formatter normalizes indentation to one unit per level; a source
    # file may legitimately use several units (e.g. two tabs) per level, so
    # derive the per-level scale from the indentation of the first line of
    # the region on both sides.
    fragment_first_indent = re.match(r"^[\t ]*", fragment_lines[0]).group(0)
    original_first_indent = re.match(
        r"^[\t ]*", original_lines[original_start - 1]
    ).group(0)
    fragment_levels = _indent_levels(fragment_first_indent, configured_unit)
    original_levels = _indent_levels(original_first_indent, source_unit)
    source_units_per_level = (
        original_levels / fragment_levels if fragment_levels else 1
    )
    fragment_lines = _reindent_fragment(
        fragment_lines, configured_unit, source_unit, source_units_per_level
    )

    fragment = "".join(fragment_lines)
    uses_crlf = any(line.endswith("\r\n") for line in original_lines)
    if uses_crlf:
        # keep the file's line ending style within the formatted fragment
        fragment = fragment.replace("\r\n", "\n").replace("\n", "\r\n")
    # Preserve presence of the line terminator of the original region;
    # everything outside it stays byte-for-byte.
    last_original_line = original_lines[original_end - 1]
    if not last_original_line.endswith(("\n", "\r")):
        if fragment.endswith("\r\n"):
            fragment = fragment[:-2]
        elif fragment.endswith("\n"):
            fragment = fragment[:-1]

    return "".join(before) + fragment + "".join(after)
