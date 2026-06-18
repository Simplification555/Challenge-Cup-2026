"""Answer normalization and lightweight equivalence checks."""

from __future__ import annotations

import ast
import math
import re
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Iterable, List, Optional


@dataclass
class AnswerForms:
    raw: str
    latex: str
    canonical: str
    answer_type: str = "other"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EquivalenceResult:
    equivalent: bool
    method: str
    confidence: float
    normalized_prediction: str
    normalized_expected: str
    issues: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


UNICODE_REPLACEMENTS = {
    "−": "-",
    "–": "-",
    "—": "-",
    "×": "*",
    "·": "*",
    "÷": "/",
    "π": "pi",
    "∞": "oo",
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    "∈": " in ",
    "，": ",",
    "；": ";",
    "：": ":",
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
    "｛": "{",
    "｝": "}",
}


def _extract_math_value(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\.+$", "", s).strip()

    # 1. Extract min/max value
    min_max_match = re.search(r"(?:minimum|maximum|min|max|objective|value)\s*(?:is|value|value is)?\s*[:=]?\s*(-?\d+(?:\.\d+)?)", s, re.IGNORECASE)
    if min_max_match:
        return min_max_match.group(1).strip()

    # 2. Extract assignment value
    assign_match = re.search(
        r"(?:"
        r"\|\|[A-Za-z]\|\|"
        r"|[\u2016\u2225][A-Za-z][\u2016\u2225]"
        r"|\\\|[A-Za-z]\\\|"
        r"|[A-Za-z](?:_[0-9A-Za-z{}]+)?"
        r"|\([A-Za-z\s*,]+\)"
        r"|[A-Za-z]\s*\([A-Za-z\s*,]+\)"
        r")\s*[:=]\s*([^:=]+)$",
        s,
    )
    if assign_match:
        return assign_match.group(1).strip()

    # 3. Extract trailing value after "is"
    is_match = re.search(r"\bis\s+([^is]+)$", s, re.IGNORECASE)
    if is_match:
        return is_match.group(1).strip()

    return s


def _extract_numbers(s: str) -> List[float]:
    raw_nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    nums = []
    for x in raw_nums:
        try:
            nums.append(float(x))
        except Exception:
            pass
    return sorted(nums)


def normalize_answer(answer: Any, answer_type: str = "other") -> AnswerForms:
    raw = str(answer or "").strip()
    cleaned = strip_answer_wrappers(raw)
    latex = normalize_latex(cleaned)
    canonical = canonicalize(latex, answer_type=answer_type)
    return AnswerForms(raw=raw, latex=latex, canonical=canonical, answer_type=answer_type)


def strip_answer_wrappers(answer: str) -> str:
    text = answer.strip()
    text = re.sub(r"^```(?:json|latex|text|math)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    text = re.sub(
        r"^(?:final\s+answer|answer|\u6700\u7ec8\u7b54\u6848|\u7b54\u6848)\s*[:\uff1a]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    # Strip currency symbols
    text = re.sub(r"^[\$\uffe5\u00a5\u00a3\u20ac]\s*", "", text)
    text = re.sub(r"\s*[\$\uffe5\u00a5\u00a3\u20ac]$", "", text)

    # Strip trailing parenthesized comments like "0.0725 (approximately 7.25%)" -> "0.0725"
    match = re.search(r"(\w*)\s*\(([^)]*)\)$", text)
    if match:
        prefix_word = match.group(1).lower()
        inside = match.group(2)
        if prefix_word not in {"sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "exp", "sqrt", "min", "max", "u", "v", "f", "g", "h"}:
            if re.search(r"[a-zA-Z]", inside) and " " in inside and "=" not in inside:
                text = text[:match.start(0) + len(match.group(1))].strip()

    boxed = extract_boxed(text)
    if boxed:
        text = boxed
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1].strip()
    return text.strip()


def extract_boxed(text: str) -> str:
    for command in (r"\boxed", r"\fbox"):
        idx = text.find(command)
        if idx == -1:
            continue
        brace = text.find("{", idx)
        if brace == -1:
            continue
        extracted = _extract_braced(text, brace)
        if extracted:
            return extracted.strip()
    return ""


def _extract_braced(text: str, start: int) -> str:
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    return ""


def normalize_latex(text: str) -> str:
    value = str(text or "").strip()
    for src, dst in UNICODE_REPLACEMENTS.items():
        value = value.replace(src, dst)
    value = value.replace("\\[", "").replace("\\]", "")
    value = value.replace("\\(", "").replace("\\)", "")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\,", "").replace("\\;", "").replace("\\!", "")
    value = value.replace("$", "")
    for src, dst in {
        "\\cdot": "*",
        "\\times": "*",
        "\\div": "/",
        "\\leq": "<=",
        "\\le": "<=",
        "\\geq": ">=",
        "\\ge": ">=",
        "\\neq": "!=",
    }.items():
        value = value.replace(src, dst)

    # Normalize degree symbols
    value = value.replace("^\\circ", "").replace("^\u2218", "").replace("^\u25e6", "")
    value = value.replace("^circ", "").replace("\\circ", "").replace("\\degree", "")

    value = _normalize_common_unicode_math(value)
    value = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\operatorname\s*\{([^{}]*)\}", r"\1", value)
    value = value.replace("\\pi", "pi").replace("\\infty", "oo")
    value = _replace_latex_matrices(value)
    value = _replace_latex_frac(value)
    value = _replace_latex_sqrt(value)
    value = _replace_latex_functions(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _normalize_common_unicode_math(text: str) -> str:
    """Handle real Unicode math symbols that often appear in model answers."""

    value = str(text or "")
    value = re.sub("\u221a" + r"\s*\{([^{}]+)\}", r"sqrt(\1)", value)
    value = re.sub("\u221a" + r"\s*\(([^()]+)\)", r"sqrt(\1)", value)
    value = re.sub("\u221a" + r"\s*([0-9A-Za-z_.]+)", r"sqrt(\1)", value)
    value = re.sub(r"\^\s*\{([^{}]+)\}", r"^(\1)", value)
    value = re.sub(r"\*\*\s*\{([^{}]+)\}", r"**(\1)", value)
    value = value.replace("\u00b0", "").replace("^\u00b0", "")
    value = _replace_unicode_superscripts(value)
    return value


def _replace_unicode_superscripts(text: str) -> str:
    superscripts = str.maketrans(
        {
            "\u2070": "0",
            "\u00b9": "1",
            "\u00b2": "2",
            "\u00b3": "3",
            "\u2074": "4",
            "\u2075": "5",
            "\u2076": "6",
            "\u2077": "7",
            "\u2078": "8",
            "\u2079": "9",
            "\u207a": "+",
            "\u207b": "-",
        }
    )

    def repl(match: re.Match[str]) -> str:
        return "^" + match.group(0).translate(superscripts)

    return re.sub(r"[\u2070\u00b9\u00b2\u00b3\u2074-\u2079\u207a\u207b]+", repl, str(text or ""))


def _replace_latex_matrices(text: str) -> str:
    def matrix_repl(match: re.Match[str]) -> str:
        body = match.group("body").strip()
        rows = [row.strip() for row in re.split(r"\\\\", body) if row.strip()]
        normalized_rows = []
        for row in rows:
            cells = [cell.strip() for cell in row.split("&")]
            normalized_rows.append(",".join(cells))
        return "[[" + "],[".join(normalized_rows) + "]]"

    matrix_pattern = re.compile(
        r"\\begin\{(?P<env>p?matrix|bmatrix|Bmatrix|vmatrix|Vmatrix|smallmatrix)\}"
        r"(?P<body>.*?)"
        r"\\end\{(?P=env)\}",
        flags=re.DOTALL,
    )
    value = matrix_pattern.sub(matrix_repl, text)

    array_pattern = re.compile(
        r"\\begin\{array\}(?:\{[^{}]*\})?"
        r"(?P<body>.*?)"
        r"\\end\{array\}",
        flags=re.DOTALL,
    )
    return array_pattern.sub(matrix_repl, value)


def _replace_latex_frac(text: str) -> str:
    value = text
    previous = None
    pat1 = re.compile(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    pat2 = re.compile(r"\\(?:dfrac|tfrac|frac)\s*([0-9A-Za-z]|\\[A-Za-z]+)\s*\{([^{}]+)\}")
    pat3 = re.compile(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*([0-9A-Za-z]|\\[A-Za-z]+)")
    pat4 = re.compile(r"\\(?:dfrac|tfrac|frac)\s*([0-9A-Za-z]|\\[A-Za-z]+)\s*([0-9A-Za-z]|\\[A-Za-z]+)")

    while previous != value:
        previous = value
        value = pat1.sub(r"(\1)/(\2)", value)
        value = pat2.sub(r"(\1)/(\2)", value)
        value = pat3.sub(r"(\1)/(\2)", value)
        value = pat4.sub(r"(\1)/(\2)", value)
    return value


def _replace_latex_sqrt(text: str) -> str:
    pattern = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
    previous = None
    value = text
    while previous != value:
        previous = value
        value = pattern.sub(r"sqrt(\1)", value)
    value = re.sub(r"\\sqrt\s+([A-Za-z0-9_.]+)", r"sqrt(\1)", value)
    return value


_LATEX_FUNCTIONS = (
    "sin",
    "cos",
    "tan",
    "cot",
    "sec",
    "csc",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "log",
    "ln",
    "exp",
)


def _replace_latex_functions(text: str) -> str:
    value = text
    for name in _LATEX_FUNCTIONS:
        pattern_braced = rf"(?:\\|\b){name}\s*\{{([^{{}}]+)\}}"
        pattern_plain = rf"(?:\\|\b){name}\s+([A-Za-z0-9_.]+)"
        value = re.sub(pattern_braced, rf"{name}(\1)", value)
        value = re.sub(pattern_plain, rf"{name}(\1)", value)
        value = value.replace(f"\\{name}", name)
    return value


def canonicalize(text: str, answer_type: str = "other") -> str:
    value = normalize_latex(text)
    value = value.strip()
    value = _drop_units(value)
    value = value.replace(" ", "")
    if not value:
        return ""

    if answer_type == "choice":
        return _canonical_choice(value)
    if answer_type == "numeric":
        numeric = _canonical_number(value)
        if numeric is not None:
            return numeric
    if answer_type == "matrix":
        matrix = _canonical_matrix(value)
        if matrix is not None:
            return matrix
    if answer_type in {"tuple", "vector"}:
        brackets = ("[", "]") if answer_type == "vector" else ("(", ")")
        sequence = _canonical_sequence(value, brackets=brackets)
        if sequence is not None:
            return sequence
    if answer_type == "set":
        set_value = _canonical_set(value, force=True)
        if set_value is not None:
            return set_value
    interval = _canonical_interval(value)
    if interval is not None:
        return interval
    matrix = _canonical_matrix(value)
    if matrix is not None:
        return matrix
    if answer_type in {"formula", "other"}:
        set_value = _canonical_set(value)
        if set_value is not None:
            return set_value

    expr = _canonical_sympy(value)
    return expr if expr is not None else value.lower()


def _canonical_choice(value: str) -> str:
    text = value.strip().upper()
    text = re.sub(r"^(OPTION|CHOICE|ANSWER|ANS)\s*[:.\u3001\uff1a]?\s*", "", text)
    match = re.match(r"^\(?([A-Z])\)?(?:[.\u3001:\uff1a\s]|$)", text)
    if match:
        return match.group(1)
    letters = re.findall(r"\b([A-Z])\b", text)
    if len(letters) == 1:
        return letters[0]
    return text


def _strip_variable_assignment(s: str) -> str:
    pattern = r"^(?:[a-zA-Z](?:_[0-9a-zA-Z{}]+)?|\([a-zA-Z\s*,]+\))\s*=\s*"
    return re.sub(pattern, "", s).strip()


def _to_flat_list(s: str) -> Optional[List[str]]:
    text = s.strip()
    text = text.replace("(", "[").replace(")", "]").replace("{", "[").replace("}", "]")
    try:
        val = ast.literal_eval(text)
        if isinstance(val, list):
            if all(not isinstance(x, list) for x in val):
                return [str(x) for x in val]
            if len(val) == 1 and isinstance(val[0], list):
                row = val[0]
                if all(not isinstance(x, list) for x in row):
                    return [str(x) for x in row]
            if all(isinstance(row, list) and len(row) == 1 and not isinstance(row[0], list) for row in val):
                return [str(row[0]) for row in val]
    except Exception:
        pass
    return None


def equivalent_answers(
    prediction: Any,
    expected: Any,
    answer_type: str = "other",
    numeric_tol: float = 1e-8,
) -> EquivalenceResult:
    pred = normalize_answer(prediction, answer_type)
    exp = normalize_answer(expected, answer_type)
    issues: List[str] = []

    # 1. Direct canonical exact match
    if pred.canonical == exp.canonical:
        return EquivalenceResult(True, "canonical_exact", 1.0, pred.canonical, exp.canonical, [])

    # 2. Match after stripping leading variable assignments
    pred_stripped = _strip_variable_assignment(pred.canonical)
    exp_stripped = _strip_variable_assignment(exp.canonical)
    if pred_stripped != pred.canonical or exp_stripped != exp.canonical:
        if pred_stripped == exp_stripped:
            return EquivalenceResult(True, "canonical_exact_after_strip_assignment", 1.0, pred_stripped, exp_stripped, [])

    # 3. Try mathematical value extraction
    pred_math = _extract_math_value(pred.raw)
    exp_math = _extract_math_value(exp.raw)
    if pred_math != pred.raw or exp_math != exp.raw:
        pred_math_can = canonicalize(pred_math, answer_type)
        exp_math_can = canonicalize(exp_math, answer_type)
        if pred_math_can == exp_math_can:
            return EquivalenceResult(True, "math_value_extracted_exact", 1.0, pred_math_can, exp_math_can, [])
        pred_stripped = _strip_variable_assignment(pred_math_can)
        exp_stripped = _strip_variable_assignment(exp_math_can)
    else:
        pred_stripped = _strip_variable_assignment(pred.canonical)
        exp_stripped = _strip_variable_assignment(exp.canonical)

    # 4. Numeric close match (using stripped/extracted forms)
    pred_num = _to_float(pred_stripped)
    exp_num = _to_float(exp_stripped)
    if pred_num is not None and exp_num is not None:
        ok = math.isclose(pred_num, exp_num, rel_tol=numeric_tol, abs_tol=numeric_tol)
        return EquivalenceResult(
            ok,
            "numeric_tolerance",
            0.98 if ok else 0.0,
            pred_stripped,
            exp_stripped,
            [] if ok else ["numeric values differ"],
        )

    # 5. Sympy equivalence (using stripped/extracted forms)
    sympy_result = _sympy_equivalent(pred_stripped, exp_stripped)
    if sympy_result is not None:
        return EquivalenceResult(
            sympy_result,
            "sympy_simplify",
            0.95 if sympy_result else 0.0,
            pred_stripped,
            exp_stripped,
            [] if sympy_result else ["sympy expressions are not equivalent"],
        )

    # 6. Flat 1D Vector/Matrix/Tuple element-by-element equivalence
    flat_pred = _to_flat_list(pred.canonical)
    flat_exp = _to_flat_list(exp.canonical)
    if flat_pred is not None and flat_exp is not None and len(flat_pred) == len(flat_exp):
        all_eq = True
        for p_elem, e_elem in zip(flat_pred, flat_exp):
            if not equivalent_answers(p_elem, e_elem, answer_type="other", numeric_tol=numeric_tol).equivalent:
                all_eq = False
                break
        if all_eq:
            return EquivalenceResult(True, "flat_vector_equivalence", 1.0, pred.canonical, exp.canonical, [])

    # 7. Try multi-number comparison (e.g. for x=3, y=7, optimal=37)
    pred_nums = _extract_numbers(pred.raw)
    exp_nums = _extract_numbers(exp.raw)
    if pred_nums and exp_nums and len(pred_nums) == len(exp_nums) and len(pred_nums) >= 2:
        if all(math.isclose(p, e, rel_tol=numeric_tol, abs_tol=numeric_tol) for p, e in zip(pred_nums, exp_nums)):
            return EquivalenceResult(True, "multi_number_equivalence", 0.96, pred.canonical, exp.canonical, [])

    issues.append("no equivalence method matched")
    return EquivalenceResult(False, "none", 0.0, pred.canonical, exp.canonical, issues)


def _drop_units(value: str) -> str:
    pattern = r"(?<=\d)\s*(cm|mm|m|kg|g|s|sec|seconds?|units?|feet|foot|inches?|inch|meters?|miles?|yards?|hours?|minutes?|days?|years?|cents?|dollars?|degrees?|\u03a9|\\Omega|ohms?)$"
    return re.sub(pattern, "", value, flags=re.I)


def _canonical_number(value: str) -> Optional[str]:
    number = _to_float(value)
    if number is None:
        return None
    if not math.isfinite(number):
        return None
    if abs(number - round(number)) < 1e-12:
        return str(int(round(number)))
    return f"{number:.12g}"


def _to_float(value: str) -> Optional[float]:
    text = value.strip()
    is_percent = False
    if text.endswith("%") or text.endswith("percent"):
        is_percent = True
        text = re.sub(r"(?:%|percent)$", "", text).strip()

    if "," in text:
        stripped = text.replace(",", "")
        try:
            val = float(stripped)
            return val / 100.0 if is_percent else val
        except ValueError:
            pass

    try:
        val = float(text)
        return val / 100.0 if is_percent else val
    except Exception:
        pass

    safe = _safe_numeric_eval(text)
    if safe is not None:
        return safe / 100.0 if is_percent else safe

    try:
        import sympy as sp
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            expr = sp.sympify(_sympy_ready(text))
        if expr.is_number:
            val = float(expr.evalf())
            if not math.isfinite(val):
                return None
            return val / 100.0 if is_percent else val
    except Exception:
        return None
    return None


def _safe_numeric_eval(value: str) -> Optional[float]:
    text = value.replace("^", "**").replace("pi", str(math.pi)).replace("oo", "inf")
    if not re.fullmatch(r"[0-9eE+\-*/()., inf]+", text):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text, mode="eval")
        return float(_eval_numeric_ast(tree.body))
    except Exception:
        return None


def _eval_numeric_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id == "inf":
        return math.inf
    if isinstance(node, ast.UnaryOp):
        value = _eval_numeric_ast(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    if isinstance(node, ast.BinOp):
        left = _eval_numeric_ast(node.left)
        right = _eval_numeric_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
    raise ValueError("unsupported numeric expression")


def _canonical_set(value: str, force: bool = False) -> Optional[str]:
    brackets = [("{", "}")]
    if force:
        brackets.extend([("[", "]"), ("(", ")")])
    wrapped = next(
        ((left, right) for left, right in brackets if value.startswith(left) and value.endswith(right)),
        None,
    )
    if not (wrapped or "," in value):
        return None
    inner = value[1:-1] if wrapped else value
    parts = _split_top_level(inner, ",")
    if len(parts) <= 1:
        return None
    canonical_parts = [canonicalize(part, "formula") for part in parts if part]
    return "{" + ",".join(sorted(canonical_parts)) + "}"


def _canonical_sequence(value: str, brackets: tuple[str, str] = ("(", ")")) -> Optional[str]:
    left, right = brackets
    inner = value
    for start, end in (("(", ")"), ("[", "]"), ("{", "}")):
        if value.startswith(start) and value.endswith(end):
            inner = value[1:-1]
            break
    parts = _split_top_level(inner, ",")
    if len(parts) <= 1:
        return None
    canonical_parts = [canonicalize(part, "formula") for part in parts if part]
    return left + ",".join(canonical_parts) + right


def _canonical_interval(value: str) -> Optional[str]:
    if len(value) < 5:
        return None
    if value[0] not in "[(" or value[-1] not in "])":
        return None
    inner = value[1:-1]
    parts = _split_top_level(inner, ",")
    if len(parts) != 2:
        return None
    left = canonicalize(parts[0], "formula")
    right = canonicalize(parts[1], "formula")
    return f"{value[0]}{left},{right}{value[-1]}"


def _canonical_matrix(value: str) -> Optional[str]:
    if not (value.startswith("[[") and value.endswith("]]")):
        return None
    rows = value[2:-2].split("],[")
    normalized_rows = []
    for row in rows:
        cells = _split_top_level(row, ",")
        normalized_rows.append(",".join(canonicalize(cell, "formula") for cell in cells))
    return "[[" + "],[".join(normalized_rows) + "]]"


def _canonical_sympy(value: str) -> Optional[str]:
    try:
        import sympy as sp

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            expr = sp.sympify(_sympy_ready(value))
        return str(sp.factor(sp.simplify(expr)))
    except Exception:
        return None


def _sympy_equivalent(left: str, right: str) -> Optional[bool]:
    try:
        import sympy as sp

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            a = sp.sympify(_sympy_ready(left))
            b = sp.sympify(_sympy_ready(right))
        return bool(sp.simplify(a - b) == 0)
    except Exception:
        return None


def _sympy_ready(value: str) -> str:
    text = value.replace("^", "**")
    text = text.replace("oo", "sp.oo")
    # sympify does not know the sp namespace here, so map back after protecting words.
    text = text.replace("sp.oo", "oo")
    text = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", text)
    text = re.sub(r"(?<=[A-Za-z)])(?=\d)", "*", text)
    text = re.sub(r"\)(?=[A-Za-z(])", ")*", text)
    return text


def _split_top_level(text: str, sep: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == sep and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return parts
