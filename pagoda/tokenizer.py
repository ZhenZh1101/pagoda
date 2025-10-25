"""Byte-level BPE tokenizer, plan section 5.

Trained with rustbpe, run with tiktoken. 32768 ids in total = BPE ranks + 17
special tokens (9 named + 8 reserved). The pre-tokenizer is GPT-4's regex with
`\\p{N}{1,3}` replaced by `\\p{N}`, so every digit is its own token.

Ordinary text is always encoded with `encode_ordinary`, so special-token
strings that appear in data ("<|bos|>") are split into normal pieces and can
never be forged.
"""

import copy
import os
import pickle
from functools import lru_cache

from pagoda.common import TOKENIZER_DIR

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>", "<|user_end|>",
    "<|assistant_start|>", "<|assistant_end|>",
    "<|python_start|>", "<|python_end|>",
    "<|output_start|>", "<|output_end|>",
] + [f"<|reserved_{i}|>" for i in range(8)]

SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

VOCAB_SIZE = 32768


class Tokenizer:
    def __init__(self, enc):
        self.enc = enc
        self.bos_id = self.special("<|bos|>")

    # --- construction -------------------------------------------------------
    @classmethod
    def train(cls, text_iterator, vocab_size=VOCAB_SIZE):
        import rustbpe
        import tiktoken

        n_merge_vocab = vocab_size - len(SPECIAL_TOKENS)
        bpe = rustbpe.Tokenizer()
        bpe.train_from_iterator(text_iterator, n_merge_vocab, pattern=SPLIT_PATTERN)
        ranks = {bytes(k): v for k, v in bpe.get_mergeable_ranks()}
        offset = len(ranks)
        enc = tiktoken.Encoding(
            name="pagoda32k",
            pat_str=bpe.get_pattern(),
            mergeable_ranks=ranks,
            special_tokens={name: offset + i for i, name in enumerate(SPECIAL_TOKENS)},
        )
        return cls(enc)

    @classmethod
    def load(cls, directory=TOKENIZER_DIR):
        path = os.path.join(directory, "tokenizer.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found; run `python -m scripts.tok_train` first")
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    def save(self, directory=TOKENIZER_DIR):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "tokenizer.pkl"), "wb") as f:
            pickle.dump(self.enc, f)

    # --- basic API ------------------------------------------------------------
    @property
    def vocab_size(self):
        return self.enc.n_vocab

    @lru_cache(maxsize=64)
    def special(self, name):
        return self.enc.encode_single_token(name)

    def encode(self, text, bos=False):
        ids = self.enc.encode_ordinary(text)
        return [self.bos_id] + ids if bos else ids

    def encode_batch(self, texts, bos=False, num_threads=8):
        out = self.enc.encode_ordinary_batch(texts, num_threads=num_threads)
        if bos:
            out = [[self.bos_id] + ids for ids in out]
        return out

    def decode(self, ids):
        return self.enc.decode(list(ids))

    def token_bytes(self):
        """Byte length of every id (0 for special tokens), used for bits-per-byte."""
        import numpy as np
        special_ids = {self.special(name) for name in SPECIAL_TOKENS}
        out = np.zeros(self.vocab_size, dtype=np.int32)
        for i in range(self.vocab_size):
            if i not in special_ids:
                out[i] = len(self.enc.decode_single_token_bytes(i))
        return out

    # --- chat rendering (plan section 9 "格式") ---------------------------------
    def render_conversation(self, messages, max_tokens=None):
        """messages: [{"role": "user"|"assistant", "content": str | [parts]}].

        Assistant content can be a list of parts {"type": "text"|"python"|"python_output", "text": ...}.
        Returns (ids, mask) where mask=1 marks tokens the assistant is trained on.
        Tool outputs are never supervised (they are injected by the executor).
        """
        sp = self.special
        ids, mask = [self.bos_id], [0]

        def add(toks, m):
            if isinstance(toks, int):
                toks = [toks]
            ids.extend(toks)
            mask.extend([m] * len(toks))

        messages = merge_system(messages)
        for i, msg in enumerate(messages):
            want = "user" if i % 2 == 0 else "assistant"
            if msg["role"] != want:
                raise ValueError(f"message {i} has role {msg['role']}, expected {want}")
            content = msg["content"]
            if want == "user":
                add(sp("<|user_start|>"), 0)
                add(self.encode(content), 0)
                add(sp("<|user_end|>"), 0)
                continue
            add(sp("<|assistant_start|>"), 0)
            parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
            for part in parts:
                toks = self.encode(part["text"])
                if part["type"] == "text":
                    add(toks, 1)
                elif part["type"] == "python":
                    add(sp("<|python_start|>"), 1)
                    add(toks, 1)
                    add(sp("<|python_end|>"), 1)
                elif part["type"] == "python_output":
                    add(sp("<|output_start|>"), 0)
                    add(toks, 0)
                    add(sp("<|output_end|>"), 0)
                else:
                    raise ValueError(f"unknown part type {part['type']}")
            add(sp("<|assistant_end|>"), 1)
        if max_tokens is not None:
            ids, mask = ids[:max_tokens], mask[:max_tokens]
        return ids, mask

    def render_for_completion(self, messages):
        """Prompt ids ending in <|assistant_start|> (the last message must be the user's)."""
        messages = merge_system(messages)
        if messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        ids, _ = self.render_conversation(messages)
        return ids + [self.special("<|assistant_start|>")]


def merge_system(messages):
    """Fold a leading system message into the first user turn."""
    if messages and messages[0]["role"] == "system":
        messages = copy.deepcopy(messages)
        sys_msg = messages.pop(0)["content"].strip()
        if messages and messages[0]["role"] == "user" and sys_msg:
            messages[0]["content"] = sys_msg + "\n\n" + messages[0]["content"]
    return messages


_TOKENIZER = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = Tokenizer.load()
    return _TOKENIZER
