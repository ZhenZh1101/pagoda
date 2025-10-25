"""Python executor for the math tool (plan section 11, change #5).

run_python(code):
  * `python -I -S -B main.py` in a fresh empty dir under sandbox/, stdin closed
  * macOS `sandbox-exec` profile: no network, no file writes outside that dir
  * 2 s wall-clock timeout (whole process group killed), CPU rlimit 3 s
  * 512 MB memory cap: macOS does not enforce RLIMIT_AS / `ulimit -v`, so a
    watchdog polls the child's RSS and kills it above the cap
  * output truncated to 256 chars; on an exception the error line is returned
    verbatim so the model can see it and retry

Also: answer parsing / numeric comparison, and `fast_exec_many` for bulk
filtering of millions of synthetic snippets (TinyGSM) in a process pool after an
AST allowlist check; the final SFT data is always re-verified with run_python.
"""

import ast
import io
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout
from dataclasses import dataclass

from pagoda.common import SANDBOX_DIR

TIMEOUT_S = 2.0
MEM_LIMIT = 512 * 1024 * 1024
MAX_OUTPUT = 256
PYTHON = sys.executable

_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{workdir}") (literal "/dev/null") (literal "/dev/stdout") (literal "/dev/stderr")
                   (literal "/dev/tty") (literal "/dev/dtracehelper"))
"""


@dataclass
class ExecResult:
    output: str          # what is fed back between <|output_start|> and <|output_end|>
    ok: bool             # exited 0 without timeout / memory kill
    timed_out: bool = False
    stdout: str = ""
    stderr: str = ""


def _limits():
    os.setsid()
    resource.setrlimit(resource.RLIMIT_CPU, (3, 4))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _kill(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _error_line(stderr):
    lines = [l for l in stderr.strip().splitlines() if l.strip()]
    return lines[-1].strip() if lines else "Error"


def truncate(s, n=MAX_OUTPUT):
    return s if len(s) <= n else s[: n - 3] + "..."


def run_python(code, timeout=TIMEOUT_S, mem_limit=MEM_LIMIT, use_sandbox_exec=True):
    import psutil

    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    workdir = os.path.realpath(tempfile.mkdtemp(prefix="run_", dir=SANDBOX_DIR))
    try:
        with open(os.path.join(workdir, "main.py"), "w") as f:
            f.write(code)
        cmd = [PYTHON, "-I", "-S", "-B", "main.py"]
        if use_sandbox_exec and os.path.exists("/usr/bin/sandbox-exec"):
            cmd = ["/usr/bin/sandbox-exec", "-p", _PROFILE.format(workdir=workdir)] + cmd
        env = {"PATH": "/usr/bin:/bin", "PYTHONIOENCODING": "utf-8", "HOME": workdir, "TMPDIR": workdir}
        proc = subprocess.Popen(cmd, cwd=workdir, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env=env, preexec_fn=_limits)
        deadline = time.monotonic() + timeout
        timed_out = mem_killed = False
        ps = None
        while proc.poll() is None:
            if time.monotonic() > deadline:
                timed_out = True
                _kill(proc)
                break
            try:
                ps = ps or psutil.Process(proc.pid)
                rss = sum(p.memory_info().rss for p in [ps] + ps.children(recursive=True))
                if rss > mem_limit:
                    mem_killed = True
                    _kill(proc)
                    break
            except psutil.Error:
                pass
            time.sleep(0.005)
        out, err = proc.communicate()
        out = out.decode("utf-8", "replace")
        err = err.decode("utf-8", "replace")
        if timed_out:
            return ExecResult(f"TimeoutError: execution exceeded {timeout:g} seconds", False, True, out, err)
        if mem_killed:
            return ExecResult("MemoryError: exceeded 512 MB", False, False, out, err)
        if proc.returncode != 0:
            msg = _error_line(err)
            text = (out.strip() + "\n" + msg) if out.strip() else msg
            return ExecResult(truncate(text), False, False, out, err)
        return ExecResult(truncate(out.strip()), True, False, out, err)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# --------------------------------------------------------------------- answers
_ANSWER_RE = re.compile(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+(?:[eE][-+]?\d+)?)")
_NUMBER_RE = re.compile(r"^-?\d[\d,]*(\.\d+)?([eE][-+]?\d+)?$|^-?\.\d+$")


def parse_number(s):
    if s is None:
        return None
    s = str(s).strip().replace(",", "").rstrip(".")
    if s.startswith("$"):
        s = s[1:]
    try:
        return float(s)
    except ValueError:
        return None


def extract_answer(text):
    """Number after the last 'Answer:' in a completion."""
    m = _ANSWER_RE.findall(text)
    return parse_number(m[-1]) if m else None


def numbers_equal(a, b, tol=1e-6):
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(1.0, abs(b))


def single_number(output):
    """The printed output if it is exactly one number, else None."""
    s = output.strip()
    if not s or "\n" in s:
        return None
    s2 = s.replace(",", "")
    return parse_number(s2) if _NUMBER_RE.match(s2) else None


def format_number(x):
    if x is None:
        return ""
    if abs(x - round(x)) < 1e-9 and abs(x) < 1e15:
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


# ------------------------------------------------------------ bulk execution
SAFE_IMPORTS = {"math", "fractions", "decimal", "datetime", "itertools", "functools", "statistics",
                "collections", "string", "operator", "re"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "exit", "quit",
                "globals", "locals", "vars", "getattr", "setattr", "delattr", "memoryview", "help"}


def is_safe_code(code):
    """Static allowlist used before in-process execution and as a data-quality filter."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] not in SAFE_IMPORTS for a in node.names):
                return False
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in SAFE_IMPORTS or node.level:
                return False
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return False
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Pow, ast.LShift)):
            # huge integer powers run in C and cannot be interrupted by the alarm
            if not (isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float))
                    and abs(node.right.value) <= 64):
                return False
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.Global, ast.Nonlocal, ast.Lambda)):
            return False
    return True


def _alarm(signum, frame):
    raise TimeoutError("timeout")


def _pool_init():
    os.chdir(tempfile.mkdtemp(prefix="bulk_", dir=SANDBOX_DIR))
    signal.signal(signal.SIGALRM, _alarm)


def _exec_one(code):
    if not is_safe_code(code):
        return None
    buf = io.StringIO()
    signal.setitimer(signal.ITIMER_REAL, 0.5)
    try:
        with redirect_stdout(buf):
            exec(compile(code, "<snippet>", "exec"), {"__name__": "__main__"})
    except BaseException:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    out = buf.getvalue()
    return out if len(out) <= 4096 else None


def _run_isolated(code):
    if not is_safe_code(code):
        return None
    r = run_python(code, timeout=1.0)
    return r.stdout if r.ok and len(r.stdout) <= 4096 else None


def fast_exec_many(codes, workers=None, chunksize=256, batch=20000, batch_timeout=600):
    """stdout of each snippet (None on error / unsafe / timeout). Pure-arithmetic code only.

    Snippets run in-process inside a spawn pool. If a batch hangs (an operation the
    alarm cannot interrupt), the pool is killed and that batch is redone one snippet
    at a time in the subprocess sandbox.
    """
    import multiprocessing as mp
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    n_workers = workers or max(1, os.cpu_count() - 2)
    results = []
    pool = ctx.Pool(n_workers, initializer=_pool_init, maxtasksperchild=20000)
    try:
        for i in range(0, len(codes), batch):
            part = codes[i: i + batch]
            try:
                results.extend(pool.map_async(_exec_one, part, chunksize=chunksize).get(timeout=batch_timeout))
            except mp.TimeoutError:
                pool.terminate()
                pool = ctx.Pool(n_workers, initializer=_pool_init, maxtasksperchild=20000)
                results.extend(_run_isolated(c) for c in part)
    finally:
        pool.terminate()
        for d in SANDBOX_DIR.glob("bulk_*"):  # worker scratch dirs
            shutil.rmtree(d, ignore_errors=True)
    return results
