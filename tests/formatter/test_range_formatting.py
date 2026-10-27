import pytest

from gdtoolkit.formatter import format_code
from gdtoolkit.formatter.range_format import format_code_in_range
from gdtoolkit.parser import parser

MAX_LINE_LENGTH = 100


def _outcome(code, start, end, **kwargs):
    return format_code_in_range(
        code, start, end, max_line_length=MAX_LINE_LENGTH, **kwargs
    )


def _assert_valid_gdscript(code):
    parser.parse(code, gather_metadata=True)


def _assert_outside_unchanged(original, result, start, end):
    original_lines = original.splitlines(keepends=True)
    result_lines = result.splitlines(keepends=True)
    assert "".join(original_lines[: start - 1]) == "".join(
        result_lines[: start - 1]
    )
    assert "".join(original_lines[end:]) == "".join(
        result_lines[len(result_lines) - (len(original_lines) - end) :]
    )


def _assert_stable(result, formatted_range, **kwargs):
    # Re-running for every single line of the formatted fragment must be a
    # no-op (repeated execution is stable).
    for line in range(formatted_range.start, formatted_range.end + 1):
        repeated = format_code_in_range(
            result,
            line,
            line,
            max_line_length=MAX_LINE_LENGTH,
            **kwargs,
        )
        assert repeated.code == result
        parser.parse(repeated.code)


def _check(code, start, end, expected_fragment=None, **kwargs):
    outcome = _outcome(code, start, end, **kwargs)
    assert outcome.formatted_range is not None
    _assert_valid_gdscript(outcome.code)
    _assert_outside_unchanged(
        code, outcome.code, outcome.original_range.start, outcome.original_range.end
    )
    _assert_stable(outcome.code, outcome.formatted_range, **kwargs)
    if expected_fragment is not None:
        lines = outcome.code.splitlines()
        fragment = lines[
            outcome.formatted_range.start - 1 : outcome.formatted_range.end
        ]
        assert fragment == expected_fragment
    return outcome


def test_simple_statement_is_the_minimal_unit():
    code = "extends Node\n\nfunc foo():\n\tx=1\n\ty=2\n"
    outcome = _check(code, 4, 4, ["\tx = 1"])
    assert (outcome.original_range.start, outcome.original_range.end) == (4, 4)
    assert outcome.code.splitlines()[4] == "\ty=2"  # outside, untouched


def test_selection_expands_from_continuation_lines_to_whole_statement():
    code = (
        "func foo():\n"
        "\tif cond:\n"
        "\t\tbar( a ,\n"
        "\t\t\tb )\n"
        "\t\tbaz()\n"
    )
    outcome = _check(code, 3, 4, ["\t\tbar(a, b)"])
    # the unit starts at the statement header (line 3), not the picked line
    assert outcome.original_range.start == 3
    assert outcome.original_range.end == 4
    # the following sibling statement is outside the unit
    assert "\t\tbaz()" in outcome.code
    _assert_outside_unchanged(code, outcome.code, 3, 4)


def test_compound_header_selection_expands_to_whole_block():
    code = (
        "func foo():\n"
        "\tif cond:\n"
        "\t\ta()\n"
        "\t\tb( )\n"
        "\telse:\n"
        "\t\tc()\n"
        "\tafter()\n"
    )
    outcome = _check(code, 2, 2)
    assert (outcome.original_range.start, outcome.original_range.end) == (2, 6)
    lines = outcome.code.splitlines()
    assert "\tb()" not in "\n".join(lines[:1])
    assert lines[6] == "\tafter()"  # statement following the if is untouched
    assert lines[3] == "\t\tb()"


def test_selection_starting_on_header_plus_body_formats_whole_compound():
    code = "func foo():\n\tif cond:\n\t\ta( )\n\t\tb( )\n\tdone( )\n"
    outcome = _check(code, 2, 3)
    assert (outcome.original_range.start, outcome.original_range.end) == (2, 4)
    assert outcome.code.splitlines()[4] == "\tdone( )"


def test_body_statement_selection_does_not_change_compound_header():
    code = "func foo():\n\tif cond:\n\t\ta( )\n\t\tb( )\n"
    outcome = _check(code, 3, 3, ["\t\ta()"])
    assert outcome.original_range.start == 3
    assert outcome.code.splitlines()[1] == "\tif cond:"
    assert outcome.code.splitlines()[3] == "\t\tb( )"


def test_deeply_nested_blocks_and_match_with_lambda():
    code = (
        "func f():\n"
        "\tmatch x:\n"
        "\t\t1:\n"
        "\t\t\tvar l = func():\n"
        "\t\t\t\ty( )\n"
        "\t\t2 when z:\n"
        "\t\t\ta( )\n"
        "\tprint( x )\n"
    )
    # selecting the lambda body line expands to the enclosing var statement
    outcome = _check(code, 5, 5, ["\t\t\tvar l = func(): y()"])
    assert outcome.original_range.start == 4
    assert outcome.original_range.end == 5
    # match header line -> whole match statement; the fragment shrinks and
    # the following statement keeps its (unformatted) bytes
    outcome = _check(code, 2, 2)
    assert (outcome.original_range.start, outcome.original_range.end) == (2, 7)
    assert outcome.code.splitlines()[6] == "\tprint( x )"


def test_attached_annotations_belong_to_the_unit():
    code = "@export\nvar x=1\nvar y=2\n"
    # the formatter merges the annotation onto the statement line; selecting
    # either of the two lines must keep them in one equivalence class
    outcome = _check(code, 2, 2, ["@export var x = 1"])
    assert (outcome.original_range.start, outcome.original_range.end) == (1, 2)
    assert outcome.code.splitlines()[1] == "var y=2"
    outcome = _check(code, 1, 1, ["@export var x = 1"])
    assert (outcome.original_range.start, outcome.original_range.end) == (1, 2)


def test_multiple_selected_siblings_form_a_single_group():
    code = "func foo():\n\ta( )\n\tb( )\n\tc( )\n"
    outcome = _check(code, 2, 3)
    assert (outcome.original_range.start, outcome.original_range.end) == (2, 3)
    lines = outcome.code.splitlines()
    assert lines[1] == "\ta()"
    assert lines[2] == "\tb()"
    assert lines[3] == "\tc( )"


def test_standalone_comment_between_statements_is_noop():
    code = "func foo():\n\ta()\n\t# lone comment\n\tb()\n"
    outcome = _outcome(code, 3, 3)
    assert outcome.formatted_range is None
    assert outcome.code == code


def test_blank_line_inside_body_is_noop():
    code = "func foo():\n\ta()\n\n\tb()\n"
    outcome = _outcome(code, 3, 3)
    assert outcome.formatted_range is None
    assert outcome.code == code


def test_standalone_comment_above_selected_statement_stays_untouched():
    code = "func foo():\n\ta()\n\t# comment\n\tvar x=1\n"
    outcome = _check(code, 4, 4, ["\tvar x = 1"])
    assert (outcome.original_range.start, outcome.original_range.end) == (4, 4)
    lines = outcome.code.splitlines()
    assert lines[2] == "\t# comment"
    assert lines[3] == "\tvar x = 1"


def test_inline_comment_in_continuation_is_preserved():
    code = (
        "func foo():\n"
        "\tvar x = [ 1 ,\n"
        "\t\t2 ,  # keep me\n"
        "\t\t3 ]\n"
    )
    outcome = _check(code, 2, 4)
    assert outcome.code.splitlines()[1] == "\tvar x = [1, 2, 3]  # keep me"


def test_trailing_comment_is_preserved():
    code = "func foo():\n\ta( )  # trailing\n"
    outcome = _check(code, 2, 2)
    assert outcome.code.splitlines()[1] == "\ta()  # trailing"


def test_spaces_indent_translation_matches_source_file():
    code = "func foo():\n  if True:\n    pass\n  x=1\n"
    # formatter is asked for 3-space indent while the file uses 2 spaces;
    # the formatted fragment must use the source file's indentation so that
    # splicing stays syntactically valid.
    outcome = _check(code, 4, 4, spaces_for_indent=3)
    assert outcome.code.splitlines()[3] == "  x = 1"
    outcome = _check(code, 2, 4, spaces_for_indent=3)
    lines = outcome.code.splitlines()
    assert lines[1] == "  if True:"
    assert lines[2] == "    pass"
    assert lines[3] == "  x = 1"


def test_crlf_and_no_final_newline_keep_outside_bytes():
    code = "extends Node\r\nfunc foo():\r\n\tx=1\r\nfunc bar():\r\n\ty=2\r\n"
    outcome = _check(code, 3, 3)
    assert outcome.code.startswith("extends Node\r\nfunc foo():\r\n")
    assert outcome.code.endswith("\r\nfunc bar():\r\n\ty=2\r\n")
    assert "\tx = 1\r\n" in outcome.code
    assert "\tx = 1\n" not in outcome.code.replace("\r\n", "")

    code = "extends Node\nfunc foo():\n\tx=1\n\ty=2"
    outcome = _check(code, 3, 3)
    assert outcome.code == "extends Node\nfunc foo():\n\tx = 1\n\ty=2"


def test_inline_property_delegate_expansion_is_mapped():
    code = "var p = 1: set = __set\nvar q = 2\n"
    outcome = _check(code, 1, 1)
    lines = outcome.code.splitlines()
    assert lines[0] == "var p = 1:"
    assert lines[1] == "\tset = __set"
    # the statement after the (grown) fragment stays byte-identical
    assert lines[-1] == "var q = 2"


def test_semicolon_companions_share_the_physical_line():
    code = "func foo():\n\ta( ) ;b( )\n\tc( )\n"
    # both statements occupy the selected line, so both belong to the unit;
    # the formatter splits them onto separate lines
    outcome = _check(code, 2, 2)
    lines = outcome.code.splitlines()
    assert lines[1] == "\ta()"
    assert lines[2] == "\tb()"
    assert lines[3] == "\tc( )"


def test_full_format_selection_equivalent_to_full_file_format():
    code = "extends Node\n\nfunc foo():\n\tx=1\n\ty=2\n"
    fully_formatted = format_code(code, max_line_length=MAX_LINE_LENGTH)
    outcome = _outcome(code, 1, 5)
    assert outcome.code == fully_formatted


def test_formatting_already_formatted_fragment_is_noop():
    code = "extends Node\n\nfunc foo():\n\tx = 1\n\ty=2\n"
    outcome = _outcome(code, 4, 4)
    assert outcome.code == code


def test_invalid_ranges_raise():
    code = "pass\n"
    with pytest.raises(ValueError):
        format_code_in_range(code, 0, 1, max_line_length=MAX_LINE_LENGTH)
    with pytest.raises(ValueError):
        format_code_in_range(code, 2, 1, max_line_length=MAX_LINE_LENGTH)
    with pytest.raises(ValueError):
        format_code_in_range(code, 5, 5, max_line_length=MAX_LINE_LENGTH)


def test_format_code_keyword_arguments_integration():
    code = "func foo():\n\tx=1\n\ty=2\n"
    result = format_code(
        code,
        max_line_length=MAX_LINE_LENGTH,
        range_start=2,
        range_end=2,
    )
    assert result == "func foo():\n\tx = 1\n\ty=2\n"


def test_multiline_string_inside_fragment_is_not_reindented():
    code = (
        "func f():\n"
        '    var s = """\n'
        "        hello\n"
        '    """\n'
        "    x=1\n"
    )
    outcome = _check(code, 2, 4, spaces_for_indent=4)
    assert "        hello" in outcome.code
    assert '"""' in outcome.code


def test_one_line_property_accessor_expands_with_its_header():
    # "get: pass" on one line must format as a whole property body - the
    # resolver must not descend into the body that shares the header line
    code = "var p:\n\tget: pass\nvar q=1\n"
    outcome = _check(code, 2, 2, ["\tget:", "\t\tpass"])
    assert outcome.original_range.start == 2
    assert outcome.code.splitlines()[3] == "var q=1"  # outside untouched


def test_one_line_property_setter_with_arguments_expands():
    code = "var p:\n\tset(_x): pass\n"
    _check(code, 2, 2, ["\tset(_x):", "\t\tpass"])


def test_repeated_structurally_identical_annotations_are_not_confused():
    # Lark trees compare structurally; identity-based indexing must keep
    # identical @abstract annotations apart
    code = (
        "@abstract\n"
        "class_name BaseClass\n\n"
        "@abstract\n"
        "class TestClass:\n"
        "\t@abstract\n"
        "\tfunc test_func()\n"
    )
    outcome = _check(code, 5, 5)
    assert outcome.original_range.start == 4
    # the unrelated top-of-file annotation + classname stay untouched
    assert outcome.code.splitlines()[0] == "@abstract"
    assert outcome.code.splitlines()[1] == "class_name BaseClass"


def test_wide_source_indentation_is_preserved_inside_fragment():
    # the file indents one nesting level with two tabs; the formatted
    # fragment must be scaled to that width so the result keeps parsing
    code = (
        "func foo(x):\n"
        "\t\tvar path: String = "
        "x.result.path.substr(0, 10).replace(\"res://\", \"\")"
        ".capitalize(1,2,3,4,6,7,8,9,10)\n"
        "\t\tvar next = 1\n"
    )
    outcome = _check(code, 2, 2)
    lines = outcome.code.splitlines()
    assert lines[1].startswith("\t\tvar path")
    assert lines[-1] == "\t\tvar next = 1"


def test_repeated_range_format_of_whole_fragment_is_stable():
    code = "func foo():\n\tx=1\n\ty=2\n\tif z:\n\t\ta( )\n"
    outcome = _outcome(code, 2, 5)
    repeated = _outcome(
        outcome.code,
        outcome.formatted_range.start,
        outcome.formatted_range.end,
    )
    assert repeated.code == outcome.code
