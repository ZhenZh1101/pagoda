"""Turn math datasets into the tool protocol of plan section 11.

  TinyGSM:  `def simple_math_problem(): ... return x`  ->  body with `print(x)`
  GSM8K:    `<<48/2=24>>` calculator annotations       ->  one assignment per annotation
  OMI-1:    <llm-code> REPL-style code                  ->  last bare expression wrapped in print()

Every converted program must print exactly one number equal to the reference
answer; callers execute it and drop the sample otherwise.
"""

import ast
import warnings
import re
import textwrap

from pagoda.sandbox import format_number, parse_number

warnings.filterwarnings("ignore", category=SyntaxWarning)

# ---------------------------------------------------------------- TinyGSM


def tinygsm_to_script(code):
    """Strip the function shell; the final `return expr` becomes `print(expr)`."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    others = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
    if len(fns) != 1 or others or fns[0].args.args or fns[0].decorator_list:
        return None
    fn = fns[0]
    body = list(fn.body)
    first_line = body[0].lineno
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        first_line = body[0].end_lineno + 1
        body = body[1:]
    if not body or not isinstance(body[-1], ast.Return) or body[-1].value is None:
        return None
    if any(isinstance(n, (ast.Return, ast.Yield, ast.YieldFrom)) for stmt in body[:-1] for n in ast.walk(stmt)):
        return None
    lines = code.split("\n")
    ret = body[-1]
    head = lines[first_line - 1: ret.lineno - 1]
    indent = re.match(r"\s*", lines[ret.lineno - 1]).group()
    expr = ast.get_source_segment(code, ret.value)
    if expr is None:
        return None
    script_body = textwrap.dedent("\n".join(head + [f"{indent}print({expr})"])).strip("\n")
    imports = [ast.get_source_segment(code, n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    script = "\n".join(imports + [script_body]) if imports else script_body
    # drop blank lines left by the docstring and trailing whitespace
    script = "\n".join(l.rstrip() for l in script.split("\n"))
    script = re.sub(r"\n{3,}", "\n\n", script).strip()
    try:
        ast.parse(script)
    except SyntaxError:
        return None
    return script


# ---------------------------------------------------------------- GSM8K

_CALC = re.compile(r"<<([^<>=]+)=([^<>]*)>>")
_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")


def gsm8k_final(answer):
    return parse_number(answer.split("####")[-1])


def gsm8k_cot(answer):
    """Readable CoT: annotations removed, '#### x' -> 'Answer: x'."""
    body, final = answer.split("####")
    body = _CALC.sub("", body).strip()
    return f"{body}\nAnswer: {format_number(parse_number(final))}"


def gsm8k_to_script(answer):
    """One `vN = expr` per calculator annotation, the sentence as a comment, then print."""
    body = answer.split("####")[0]
    out, results, last = [], {}, None
    n = 0
    for line in body.strip().split("\n"):
        calcs = _CALC.findall(line)
        if not calcs:
            continue
        sentence = _CALC.sub("", line).strip()
        if sentence:
            out.append(f"# {sentence}")
        for expr, res in calcs:
            expr = expr.replace("$", "").replace(",", "").replace("x", "*").replace("÷", "/").strip()
            if not re.fullmatch(r"[\d\s.+\-*/()%]+", expr):
                return None
            n += 1
            name = f"v{n}"

            def sub(m):
                v = parse_number(m.group())
                return results.get(v, m.group())

            out.append(f"{name} = {_NUM.sub(sub, expr)}")
            rv = parse_number(res)
            if rv is not None:
                results[rv] = name
            last = name
    if last is None:
        return None
    out.append(f"print({last})")
    return "\n".join(out)


# ---------------------------------------------------------------- OpenMathInstruct-1

_LLM_CODE = re.compile(r"(?s)^(.*?)<llm-code>\n?(.*?)</llm-code>\s*<llm-code-output>\n?(.*?)</llm-code-output>(.*)$")
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def omi1_to_parts(solution):
    """-> (analysis_text, script, closing_text) or None."""
    if solution.count("<llm-code>") != 1:
        return None
    m = _LLM_CODE.match(solution)
    if not m:
        return None
    pre, code, _, post = m.groups()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    if not tree.body:
        return None
    last = tree.body[-1]
    lines = code.rstrip("\n").split("\n")
    if isinstance(last, ast.Expr) and not (isinstance(last.value, ast.Call)
                                           and getattr(last.value.func, "id", "") == "print"):
        expr = ast.get_source_segment(code, last.value)
        lines = lines[: last.lineno - 1] + [f"print({expr})"]
    script = "\n".join(lines).strip()
    post = _BOXED.sub(lambda mm: mm.group(1), post).strip()
    pre = pre.strip()
    return pre, script, post


# ---------------------------------------------------------------- OpenMathInstruct-2 (text CoT)


def omi2_cot(solution, expected):
    """Replace \\boxed{x} by x and end with 'Answer: x' (numeric answers only)."""
    ans = parse_number(expected)
    if ans is None:
        return None
    text = _BOXED.sub(lambda mm: mm.group(1), solution).strip()
    return f"{text}\nAnswer: {format_number(ans)}"


# ---------------------------------------------------------------- assistant parts


def tool_parts(script, output, answer, analysis="", closing=""):
    """Assistant content for the tool protocol (render with Tokenizer.render_conversation)."""
    parts = []
    if analysis:
        parts.append({"type": "text", "text": analysis.strip() + "\n"})
    parts.append({"type": "python", "text": "\n" + script.strip() + "\n"})
    parts.append({"type": "python_output", "text": output.strip()})
    tail = (closing.strip() + "\n") if closing else ""
    parts.append({"type": "text", "text": "\n" + tail + f"Answer: {answer}"})
    return parts
