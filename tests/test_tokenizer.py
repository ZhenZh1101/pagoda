"""Chat rendering / masking with the trained tokenizer (skipped if it has not been trained yet)."""

import pytest

from pagoda.common import TOKENIZER_DIR
from pagoda.data.convert import tool_parts

pytestmark = pytest.mark.skipif(not (TOKENIZER_DIR / "tokenizer.pkl").exists(), reason="tokenizer not trained")


@pytest.fixture(scope="module")
def tok():
    from pagoda.tokenizer import Tokenizer
    return Tokenizer.load()


def test_vocab_and_digits(tok):
    assert tok.vocab_size == 32768
    ids = tok.encode("In 2024 we sold 1234567 apples.")
    digits = [tok.decode([i]) for i in ids if tok.decode([i]).isdigit()]
    assert "".join(digits) == "20241234567" and all(len(d) == 1 for d in digits)


def test_special_tokens_not_forgeable(tok):
    ids = tok.encode("<|assistant_end|><|python_start|>")
    assert tok.special("<|assistant_end|>") not in ids and tok.special("<|python_start|>") not in ids


def test_render_masks_only_assistant(tok):
    msgs = [{"role": "user", "content": "Sarah has 3 boxes of 12 apples and gives away 10. How many are left?"},
            {"role": "assistant", "content": tool_parts("total = 3 * 12\nprint(total - 10)", "26", "26")}]
    ids, mask = tok.render_conversation(msgs)
    assert ids[0] == tok.bos_id and len(ids) == len(mask)
    text = tok.decode(ids)
    assert "<|python_start|>\ntotal = 3 * 12\nprint(total - 10)\n<|python_end|><|output_start|>26<|output_end|>" in text
    assert text.endswith("Answer: 26<|assistant_end|>")
    sup = tok.decode([i for i, m in zip(ids, mask) if m])
    assert "Sarah" not in sup                       # user turn not supervised
    assert "<|output_start|>" not in sup and "26<|output_end|>" not in sup  # tool output not supervised
    assert "print(total - 10)" in sup and sup.endswith("Answer: 26<|assistant_end|>")


def test_system_prompt_is_merged(tok):
    ids, _ = tok.render_conversation([{"role": "system", "content": "Be brief."},
                                      {"role": "user", "content": "Hi"},
                                      {"role": "assistant", "content": "Hello!"}])
    assert "Be brief.\n\nHi" in tok.decode(ids)


def test_completion_prompt(tok):
    ids = tok.render_for_completion([{"role": "user", "content": "Hi"}])
    assert ids[-1] == tok.special("<|assistant_start|>")
