import ast
import re
from pathlib import Path

_ASSERT_ATTRS = {"gridClose", "samplesClose", "momentsClose"}
_DIVIDER_RE = re.compile(r"(\S)\1{4,}")


class Snap:
    """
    One parsed view of a file: its exact text, that text split into lines, its
    AST and its token stream, all of the same instant.

    Fixers address the file by line number, so a Snap taken before an earlier
    fixer ran points at moved lines; check_py rebuilds this between fixers.
    """

    __slots__ = ("src", "lines", "tree", "toks")

    def __init__(self, src: str, tree, toks: list):
        self.src = src
        self.lines = src.splitlines()
        self.tree = tree
        self.toks = toks


class Ctx:
    __slots__ = ("path", "fixes", "errs", "msgs")

    def __init__(self, path: Path, fixes: bool):
        self.path = path
        self.fixes = fixes
        self.errs: list[str] = []
        self.msgs: list[str] = []

    def E(self, line, msg):
        self.errs.append(f"{self.path}:{line}: {msg}")

    def fix(self, msg):
        self.msgs.append(f"fixed  {self.path} ({msg})")


def is_docstring(node) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def is_assert_call(node) -> bool:
    if isinstance(node, ast.Assert):
        return True

    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
        return False

    func = node.value.func
    attr = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else None)

    if attr is None:
        return False

    isCamel = attr.startswith("assert") and (len(attr) <= 6 or attr[6].isupper())

    return isCamel or attr in _ASSERT_ATTRS
