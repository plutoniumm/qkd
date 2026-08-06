import difflib
import sys
import tempfile
import textwrap
from pathlib import Path

from .lint_checks import check_py

# Regression cases for `python -m view.lint --fix`, run as
# `python -m view.lint_test`. Findings-level tests cannot catch a fixer that
# writes the wrong bytes, so these pin the bytes.
#
# Each case is (name, before, after). `after` is what one pass of --fix must
# produce exactly, so a case also asserts that nothing outside the violation
# moved. Never regenerate an `after` from the current implementation -- that
# would let a rewrite pin itself.

CASES = (
    (
        "blank_return",
        """
        import os


        def one():
            x = os.sep
            return x
        """,
        """
        import os


        def one():
            x = os.sep

            return x
        """,
    ),
    (
        "doc_inline",
        '''
        import os


        def one():
            """Text on the opening line
            """

            return os.sep
        ''',
        '''
        import os


        def one():
            """
            Text on the opening line
            """

            return os.sep
        ''',
    ),
    (
        "doc_oneline",
        '''
        import os


        def one():
            """Text and close on one line."""

            return os.sep
        ''',
        '''
        import os


        def one():
            """
            Text and close on one line.
            """

            return os.sep
        ''',
    ),
    (
        "doc_escape",
        r'''
        import os


        def one():
            """
            A tab is \t in here.
            """

            return os.sep
        ''',
        r'''
        import os


        def one():
            """
            A tab is \\t in here.
            """

            return os.sep
        ''',
    ),
    (
        "divider",
        """
        import os

        # ------------------------------
        X = os.sep
        """,
        """
        import os

        X = os.sep
        """,
    ),
    (
        "type_ignore",
        """
        import os

        X = os.sep  # type: ignore
        """,
        """
        import os

        X = os.sep
        """,
    ),
    (
        "return_after",
        """
        import os


        def one(v):
            if v:

                return 1
            y = os.sep

            return y
        """,
        """
        import os


        def one(v):
            if v:

                return 1

            y = os.sep

            return y
        """,
    ),
    (
        "doc_then_assert",
        '''
        import os


        def one():
            """Opening-line text."""

            return os.sep


        def two(v):
            """
            A docstring three
            lines long.
            """
            x = os.sep
            assert v == x, "v"
        ''',
        '''
        import os


        def one():
            """
            Opening-line text.
            """

            return os.sep


        def two(v):
            """
            A docstring three
            lines long.
            """
            x = os.sep
            assert v == x, "v"
        ''',
    ),
    (
        "doc_then_return",
        '''
        import os


        def one():
            """Opening-line text."""

            return os.sep


        def two():
            """
            A docstring three
            lines long.
            """
            y = os.sep
            return y
        ''',
        '''
        import os


        def one():
            """
            Opening-line text.
            """

            return os.sep


        def two():
            """
            A docstring three
            lines long.
            """
            y = os.sep

            return y
        ''',
    ),
    (
        "divider_then_ignore",
        """
        import os

        # ------------------------------
        X = os.sep  # type: ignore
        Y = X
        """,
        """
        import os

        X = os.sep
        Y = X
        """,
    ),
    (
        "return_then_divider",
        """
        import os


        def one():
            x = os.sep
            return x


        # ==============================
        Y = one()
        """,
        """
        import os


        def one():
            x = os.sep

            return x


        Y = one()
        """,
    ),
    (
        "doc_return_assert",
        '''
        import os


        def one():
            """Opening-line text."""
            x = os.sep
            return x


        def two(v):
            """
            A docstring three
            lines long.
            """
            y = os.sep
            assert v == y, "v"
        ''',
        '''
        import os


        def one():
            """
            Opening-line text.
            """
            x = os.sep

            return x


        def two(v):
            """
            A docstring three
            lines long.
            """
            y = os.sep
            assert v == y, "v"
        ''',
    ),
    (
        "doc_divider_ignore",
        '''
        import os


        def one():
            """Opening-line text."""

            return os.sep


        # ------------------------------
        X = one()  # type: ignore
        ''',
        '''
        import os


        def one():
            """
            Opening-line text.
            """

            return os.sep


        X = one()
        ''',
    ),
    (
        "exam_docstrings",
        '''
        import os


        class Exam:
            def test_alpha(self):
                """
                Alpha does a thing, and this cell is copied into the report
                table verbatim.
                """
                x = os.sep

                return x

            def test_beta(self):
                """Beta's summary sits on the opening line.
                """

                return os.sep

            def test_gamma(self):
                """
                Gamma does another thing, and this docstring is long enough
                that a stale insert lands inside it.
                """
                y = os.sep
                assert y, "y"
        ''',
        '''
        import os


        class Exam:
            def test_alpha(self):
                """
                Alpha does a thing, and this cell is copied into the report
                table verbatim.
                """
                x = os.sep

                return x

            def test_beta(self):
                """
                Beta's summary sits on the opening line.
                """

                return os.sep

            def test_gamma(self):
                """
                Gamma does another thing, and this docstring is long enough
                that a stale insert lands inside it.
                """
                y = os.sep
                assert y, "y"
        ''',
    ),
    (
        "clean",
        '''
        import os


        def one(v):
            """
            Nothing here is fixable, so --fix must not touch the file.
            """
            x = os.sep

            assert v == x, "v"

            return x
        ''',
        '''
        import os


        def one(v):
            """
            Nothing here is fixable, so --fix must not touch the file.
            """
            x = os.sep

            assert v == x, "v"

            return x
        ''',
    ),
)


def body(text):
    """
    Turn an indented triple-quoted fixture into file text: no leading blank
    line, no common indent, one trailing newline.
    """

    return textwrap.dedent(text).lstrip("\n")


def show(want, got):
    """
    A unified diff of want against got, indented as one block.
    """

    diff = difflib.unified_diff(
        want.splitlines(),
        got.splitlines(),
        "want",
        "got",
        lineterm="",
    )

    return "\n".join(f"    {ln}" for ln in diff)


def once(tmp, case):
    """
    Run one fixture through --fix and return the ways it went wrong: the wrong
    output, a second pass that is not a no-op, or a finding that survived.
    """

    name, before, after = case
    path = tmp / f"{name}.py"
    want = body(after)
    path.write_text(body(before))
    check_py((path, True))
    got = path.read_text()
    bad = []

    if got != want:
        bad.append(f"!! {name}: --fix wrote the wrong file\n{show(want, got)}")

    check_py((path, True))
    if path.read_text() != got:
        bad.append(f"!! {name}: a second --fix pass changed the file again")

    errs, _ = check_py((path, False))
    if errs:
        left = [e.split(": ", 1)[-1] for e in errs]
        bad.append(f"!! {name}: findings survived the fix: {left}")

    return bad


if __name__ == "__main__":
    fails = []

    with tempfile.TemporaryDirectory(prefix="lintfix-") as tmpdir:
        tmp = Path(tmpdir)
        for case in CASES:
            fails.extend(once(tmp, case))

    for line in fails:
        print(line)

    if fails:
        print(f"FAIL  {len(CASES)} --fix case(s), {len(fails)} failure(s)")
        sys.exit(1)

    print(f"ok  {len(CASES)} --fix case(s)")
