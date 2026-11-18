"""Answer extraction and equivalence checking (final answers and subgoal claims)."""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional

RESULT_RE = re.compile(r"RESULT\s*:\s*(.+)", re.I)


def last_boxed(text: str) -> Optional[str]:
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
        k += 1
    return None


def extract_result(text: str) -> Optional[str]:
    """Executor output -> its claimed value: last 'RESULT:' line, else last \\boxed{}."""
    ms = RESULT_RE.findall(text)
    if ms:
        v = ms[-1].strip()
        b = last_boxed(v)
        return (b if b is not None else v).strip().rstrip(".")
    b = last_boxed(text)
    return b.strip() if b is not None else None


_STRIP = [r"\left", r"\right", r"\!", r"\,", r"\;", r"\:", "$", r"\displaystyle", r"\text{ }", "^\\circ", "^{\\circ}"]


def normalize(s: str) -> str:
    s = s.strip()
    for t in _STRIP:
        s = s.replace(t, "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(d|t)frac", r"\\frac", s)
    s = s.replace(" ", "").rstrip(".")
    if "=" in s and s.split("=")[-1]:
        s = s.split("=")[-1]   # "x_A=245", "3+12+5=20": compare the value after the last "="
    return s


def _to_number(s: str) -> Optional[Fraction]:
    s = normalize(s).replace(",", "")
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        return Fraction(int(m.group(1)), int(m.group(2)))
    m = re.fullmatch(r"(-?\d+)/(-?\d+)", s)
    if m and int(m.group(2)) != 0:
        return Fraction(int(m.group(1)), int(m.group(2)))
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def _sympy_equal(a: str, b: str) -> bool:
    try:
        import sympy
        from sympy.parsing.latex import parse_latex  # needs antlr4; optional
    except Exception:
        return False
    try:
        ea, eb = parse_latex(a), parse_latex(b)
        return bool(sympy.simplify(ea - eb) == 0)
    except Exception:
        return False


def equivalent(pred: Optional[str], gold: Optional[str]) -> bool:
    if pred is None or gold is None:
        return False
    a, b = normalize(pred), normalize(gold)
    if a == b:
        return True
    na, nb = _to_number(a), _to_number(b)
    if na is not None and nb is not None:
        return na == nb
    return _sympy_equal(a, b)
