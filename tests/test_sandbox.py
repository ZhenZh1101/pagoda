"""The 5 executor cases from the plan's checklist: normal, timeout, exception, no output, multiple prints."""

from pagoda.sandbox import (extract_answer, fast_exec_many, format_number, is_safe_code, numbers_equal,
                            run_python, single_number)


def test_normal():
    r = run_python("total = 3 * 12\nprint(total - 10)\n")
    assert r.ok and r.output == "26"


def test_timeout():
    r = run_python("while True:\n    pass\n")
    assert not r.ok and r.timed_out and r.output.startswith("TimeoutError")


def test_exception():
    r = run_python("x = 1 / 0\n")
    assert not r.ok and r.output == "ZeroDivisionError: division by zero"


def test_no_output():
    r = run_python("x = 5\n")
    assert r.ok and r.output == ""


def test_multiple_prints():
    r = run_python("print(1)\nprint(2)\n")
    assert r.ok and r.output == "1\n2"
    assert single_number(r.output) is None


def test_output_truncated():
    r = run_python("print('a' * 1000)")
    assert len(r.output) == 256


def test_no_network_no_writes_outside():
    r = run_python("import socket\ns = socket.create_connection(('1.1.1.1', 80), timeout=1)\nprint('connected')")
    assert not r.ok and "connected" not in r.output
    r = run_python("open('/tmp/pagoda_sandbox_escape.txt', 'w').write('x')\nprint('wrote')")
    assert not r.ok and "wrote" not in r.output


def test_memory_cap():
    r = run_python("x = bytearray(900 * 1024 * 1024)\nfor i in range(0, len(x), 4096): x[i] = 1\nprint('done')", timeout=5)
    assert not r.ok and "done" not in r.output


def test_answer_parsing():
    assert extract_answer("blah\nAnswer: 26") == 26
    assert numbers_equal(extract_answer("Answer: 26.0"), 26)
    assert extract_answer("Answer: $1,234.50") == 1234.5
    assert extract_answer("no answer here") is None
    assert format_number(26.0) == "26" and format_number(2.5) == "2.5"
    assert single_number("26.0") == 26.0 and single_number("26 apples") is None


def test_fast_exec_many():
    codes = ["print(2 + 3)", "import os\nprint(os.getcwd())", "x = 1/0", "print(10 ** 100000)"]
    assert not is_safe_code(codes[1]) and not is_safe_code(codes[3])
    out = fast_exec_many(codes, workers=2)
    assert out[0] == "5\n" and out[1] is None and out[2] is None and out[3] is None
