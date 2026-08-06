import ast
import io
import re
import tokenize

from .lint_ctx import Ctx, Snap, _DIVIDER_RE, is_assert_call, is_docstring

_ESCAPE_RE = re.compile(r"(?<!\\)\\([nrtbfva])(?!\\)")
_HOLDERS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)

# Method names a harness discovers by prefix and then prints as report-table
# row labels: MDR's exams, ast.NodeVisitor's dispatch, MDB's trials. The caller
# fixes their shape, so the two-component limit does not apply.
_DISPATCHED = ("test_", "visit_", "bench_")


def parse(ctx: Ctx, src: str):
    """
    Build a Snap of `src`, or record the SyntaxError and give back None. A file
    that does not parse gets exactly one finding and no further checks.
    """

    try:
        tree = ast.parse(src, filename=str(ctx.path))
    except SyntaxError as exc:
        ctx.E(exc.lineno or 0, f"SyntaxError: {exc.msg}")

        return None

    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError:
        toks = []

    return Snap(src, tree, toks)


def check_py(args: tuple) -> tuple[list[str], list[str]]:
    """
    Run every check over one file, re-reading and re-parsing between them
    whenever a fixer changed the file on disk.

    The re-parse is load-bearing: seven of these checks rewrite the file and
    each addresses it by line number, so a fixer must see the offsets the
    previous fix left, not the ones the pipeline was entered with. It is paid
    only on the `--fix` path, and only when something changed.
    """

    path, fixes = args
    ctx = Ctx(path, fixes)
    snap = parse(ctx, path.read_text())

    for check in CHECKS:
        if snap is None:
            break

        check(ctx, snap)

        if not ctx.fixes:
            continue

        text = path.read_text()
        if text != snap.src:
            snap = parse(ctx, text)

    return ctx.errs, ctx.msgs


def holders(snap: Snap):
    """
    Every node that can carry a docstring, paired with the docstring it does
    carry. Nodes without one are skipped.
    """

    for node in ast.walk(snap.tree):
        if not isinstance(node, _HOLDERS):
            continue
        if node.body and is_docstring(node.body[0]):
            yield node.body[0]


def file_start(ctx: Ctx, snap: Snap):
    for i, line in enumerate(snap.lines):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#!"):
            continue
        if s.startswith("#") or s.startswith('"""') or s.startswith("'''"):
            ctx.E(i + 1, "File starts with a comment — begin with imports instead")
        break


# docstrings, doc_escapes, divider_comments and type_ignore below all follow the
# same collect-violations -> rebuild-lines -> write shape, and it stays unfactored
# on purpose: each edit is a different kind (insert-before / rewrite-in-place /
# delete-or-truncate / strip-suffix), so a shared apply would need a kind parameter
# -- in the one file whose own 2026-08-13 corruption bug was exactly a fixer
# addressing lines by number against a stale AST. lint_test.py's 15 pinned
# byte-cases are what a shared apply would have to re-prove.
def docstrings(ctx: Ctx, snap: Snap):
    violations: list[int] = []

    for ds in holders(snap):
        stripped = snap.lines[ds.lineno - 1].lstrip()
        for q in ('"""', "'''"):
            if stripped.startswith(q):
                after = stripped[len(q) :]
                if after.strip() and not after.strip().startswith(q):
                    ctx.E(ds.lineno, f"Docstring text on same line as opening {q}")
                    violations.append(ds.lineno)
                break

    if not (ctx.fixes and violations):
        return

    cur = list(snap.lines)
    for lineno in sorted(set(violations), reverse=True):
        raw = cur[lineno - 1]
        pad = " " * (len(raw) - len(raw.lstrip()))
        stripped = raw.lstrip()
        for q in ('"""', "'''"):
            if stripped.startswith(q):
                inner = stripped[len(q) :]
                if inner.rstrip().endswith(q):
                    inner = inner.rstrip()[: -len(q)].strip()
                    cur[lineno - 1] = f"{pad}{q}"
                    cur.insert(lineno, f"{pad}{q}")
                    cur.insert(lineno, f"{pad}{inner}")
                else:
                    cur[lineno - 1] = f"{pad}{q}"
                    cur.insert(lineno, f"{pad}{inner.strip()}")
                break

    ctx.path.write_text("\n".join(cur) + "\n")
    ctx.fix(f"{len(set(violations))} docstring(s)")


def doc_escapes(ctx: Ctx, snap: Snap):
    ranges: list[tuple[int, int]] = []
    found = False

    for ds in holders(snap):
        ranges.append((ds.lineno, ds.end_lineno))
        chunk = "\n".join(snap.lines[ds.lineno - 1 : ds.end_lineno])
        for m in re.finditer(_ESCAPE_RE, chunk):
            found = True
            char = m.group(1)
            ctx.E(
                ds.lineno + chunk[: m.start()].count("\n"),
                f"Docstring contains \\{char}; use \\\\{char} for literal escape",
            )

    if not (ctx.fixes and found):
        return

    cur = list(snap.lines)
    for start, end in ranges:
        for i in range(start - 1, end):
            cur[i] = re.sub(_ESCAPE_RE, r"\\\\\1", cur[i])

    text = "\n".join(cur) + "\n"
    if text != snap.src:
        ctx.path.write_text(text)
        ctx.fix("docstring escapes")


def blank_before(ctx: Ctx, snap: Snap, what: str, pick):
    """
    Every node `pick` accepts needs a blank line above it. `what` names the kind
    in the message and in the fix log.
    """

    violations: list[int] = []

    for node in ast.walk(snap.tree):
        if not pick(node):
            continue

        ln = node.lineno
        if ln < 2:
            continue

        prev = snap.lines[ln - 2].strip()

        if not prev:
            continue
        if prev.endswith(":"):
            continue
        if prev.endswith('"""') or prev.endswith("'''"):
            continue
        if prev.startswith("#"):
            continue

        violations.append(ln)

    if not violations:
        return

    if ctx.fixes:
        cur = list(snap.lines)
        for ln in sorted(set(violations), reverse=True):
            cur.insert(ln - 1, "")
        ctx.path.write_text("\n".join(cur) + "\n")
        ctx.fix(f"{len(set(violations))} {what}(s)")
    else:
        for ln in sorted(set(violations)):
            ctx.E(ln, f"Missing blank line before {what}")


def blank_returns(ctx: Ctx, snap: Snap):
    blank_before(ctx, snap, "return", lambda n: isinstance(n, ast.Return))


def blank_asserts(ctx: Ctx, snap: Snap):
    """
    Left unregistered: at 4,699 assertion calls the rule mandated 4,136 blank
    lines, 8% of test/, for separation nobody was missing. blank_returns stays
    -- a return is a control-flow exit and the gap marks it. Re-register in
    CHECKS to bring it back.
    """


def return_after(ctx: Ctx, snap: Snap):
    violations: list[int] = []

    for node in ast.walk(snap.tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        rets = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
        if len(rets) < 2:
            continue
        for ret in sorted(rets, key=lambda r: r.end_lineno)[:-1]:
            after = ret.end_lineno
            if after >= len(snap.lines):
                continue
            nxt = snap.lines[after].strip()
            if not nxt:
                continue
            if nxt.startswith(("else", "elif", "except", "finally")):
                continue
            violations.append(ret.end_lineno)

    if not violations:
        return

    if ctx.fixes:
        cur = list(snap.lines)
        for ins in sorted(set(violations), reverse=True):
            cur.insert(ins, "")
        ctx.path.write_text("\n".join(cur) + "\n")
        ctx.fix(f"{len(set(violations))} return-after(s)")
    else:
        for ins in sorted(set(violations)):
            ctx.E(ins, "Missing blank line after return in multi-return function")


def assert_messages(ctx: Ctx, snap: Snap):
    for node in ast.walk(snap.tree):
        if isinstance(node, ast.Assert):
            if node.msg is None:
                ctx.E(node.lineno, "assert missing failure message")
            continue

        if not is_assert_call(node):
            continue

        call = node.value
        has_msg = any(kw.arg == "msg" for kw in call.keywords)
        if not has_msg and call.args:
            last = call.args[-1]
            has_msg = isinstance(last, ast.JoinedStr) or (
                isinstance(last, ast.Constant) and isinstance(last.value, str)
            )

        if not has_msg:
            ctx.E(node.lineno, "assert missing msg= argument")


def inline_dicts(ctx: Ctx, snap: Snap):
    for node in ast.walk(snap.tree):
        if not isinstance(node, ast.Dict):
            continue
        if len(node.keys) < 2:
            continue
        if node.lineno == node.end_lineno:
            ctx.E(
                node.lineno,
                f"Inline dict with {len(node.keys)} keys — expand to one key per line",
            )


def trailing_comma(ctx: Ctx, snap: Snap):
    skip = {
        tokenize.NEWLINE,
        tokenize.NL,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.COMMENT,
    }
    toks = snap.toks

    for node in ast.walk(snap.tree):
        if not isinstance(node, ast.Dict):
            continue
        if node.lineno == node.end_lineno or not node.keys:
            continue

        rbrace = next(
            (
                i
                for i, t in enumerate(toks)
                if t.start[0] == node.end_lineno and t.type == tokenize.OP and t.string == "}"
            ),
            None,
        )
        if rbrace is None:
            continue

        j = rbrace - 1
        while j >= 0 and toks[j].type in skip:
            j -= 1

        if j >= 0 and not (toks[j].type == tokenize.OP and toks[j].string == ","):
            ctx.E(
                toks[j].end[0],
                "Multi-line dict missing trailing comma after last entry",
            )


def divider_comments(ctx: Ctx, snap: Snap):
    found = [t for t in snap.toks if t.type == tokenize.COMMENT and _DIVIDER_RE.search(t.string[1:])]
    if not found:
        return

    if ctx.fixes:
        cur = list(snap.lines)
        for tok in sorted(found, reverse=True, key=lambda t: t.start[0]):
            ln = tok.start[0] - 1
            stripped = cur[ln][: tok.start[1]].rstrip()
            if stripped:
                cur[ln] = stripped
            else:
                del cur[ln]
        ctx.path.write_text("\n".join(cur) + "\n")
        ctx.fix(f"{len(found)} divider comment(s)")
    else:
        for tok in found:
            ctx.E(tok.start[0], "Divider comment — remove it")


def semicolons(ctx: Ctx, snap: Snap):
    for tok in snap.toks:
        if tok.type == tokenize.OP and tok.string == ";":
            ctx.E(
                tok.start[0],
                "Semicolon separating statements on one line — split onto separate lines",
            )


def type_ignore(ctx: Ctx, snap: Snap):
    found = [t for t in snap.toks if t.type == tokenize.COMMENT and "type: ignore" in t.string]
    if not found:
        return

    if ctx.fixes:
        cur = list(snap.lines)
        for tok in found:
            ln = tok.start[0] - 1
            cur[ln] = re.sub(r"\s*#\s*type:\s*ignore[^\n]*", "", cur[ln]).rstrip()
        ctx.path.write_text("\n".join(cur) + "\n")
        ctx.fix(f"{len(found)} type: ignore comment(s)")
    else:
        for tok in found:
            ctx.E(tok.start[0], "type: ignore comment — fix the type error instead")


def _word_count(name: str) -> int:
    core = name.strip("_")
    if not core:
        return 1

    expanded = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", core)

    return len([p for p in expanded.split("_") if p])


def variable_names(ctx: Ctx, snap: Snap):
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self._stack: list[set[str]] = []

        def visit_FunctionDef(self, node):
            params: set[str] = {a.arg for a in node.args.args + node.args.posonlyargs + node.args.kwonlyargs}
            if node.args.vararg:
                params.add(node.args.vararg.arg)
            if node.args.kwarg:
                params.add(node.args.kwarg.arg)

            self._stack.append(params)
            self.generic_visit(node)
            self._stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not child.name.startswith(_DISPATCHED) and _word_count(child.name) > 2:
                        ctx.E(
                            child.lineno,
                            f"'{child.name}' has more than 2 name components — shorten it",
                        )
                    self.visit(child)

        def visit_Name(self, node):
            if not self._stack:
                return

            if not isinstance(node.ctx, ast.Store):
                return

            name = node.id
            params = set().union(*self._stack)
            if name in params:
                return
            if _word_count(name) > 2:
                ctx.E(
                    node.lineno,
                    f"'{name}' has more than 2 name components — shorten it",
                )

    Visitor().visit(snap.tree)


# Order matters, and so does the fact that this is a list of one-argument
# callables: check_py re-parses between them, so every entry sees the file as
# the entry before it left it.
CHECKS = (
    file_start,
    docstrings,
    doc_escapes,
    blank_returns,
    return_after,
    assert_messages,
    inline_dicts,
    trailing_comma,
    divider_comments,
    semicolons,
    type_ignore,
    variable_names,
)
