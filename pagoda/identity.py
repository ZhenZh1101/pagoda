"""Who the model thinks it is: Pagoda-0.1 (identity/persona.md).

The model knows its name from three sources only, all built from identity/:

  1. the inference-time system prompt (persona.yaml: system_prompt), folded into
     the first user turn and shown on a small share of SFT conversations
  2. identity conversations (conversations.jsonl), ~1% of the stage-2 chat share
     and ~2% of SFT conversations
  3. short pretraining texts about Pagoda (pretrain.jsonl), ~1M tokens at the end
     of pretraining (stage 2)

This module loads the persona and holds the checks shared by data generation
(scripts.gen_identity), the prepare step and the acceptance eval
(scripts.identity_eval): says the name / claims to be another model / states a
wrong fact / is too close to a held-out eval prompt.
"""

import hashlib
import json
import random
import re
from functools import lru_cache

from pagoda.common import CODE_ROOT, iter_jsonl, load_yaml

IDENTITY_DIR = CODE_ROOT / "identity"
CONVERSATIONS = IDENTITY_DIR / "conversations.jsonl"
PRETRAIN_TEXTS = IDENTITY_DIR / "pretrain.jsonl"
EVAL_FILE = IDENTITY_DIR / "eval.json"

# conversation categories (scripts.gen_identity): the first two must say the name,
# in_task must never say it (the user addresses Pagoda, the reply just does the task)
CATEGORIES = ("name", "denial", "creator", "capabilities", "multiturn", "in_task", "persona", "edge")
MUST_SAY_NAME = {"name", "denial"}
NEVER_SAY_NAME = {"in_task"}

_NEG = re.compile(r"\b(not|never|no|nor|neither|isn't|wasn't|aren't|weren't|don't|didn't)\b|n't\b", re.I)


@lru_cache(maxsize=1)
def persona():
    return load_yaml(IDENTITY_DIR / "persona.yaml")


def system_prompt(rng=None, variant_frac=0.25):
    """The canonical system prompt; with an rng, a variant `variant_frac` of the time."""
    p = persona()
    if rng is not None and p.get("system_prompt_variants") and rng.random() < variant_frac:
        return rng.choice(p["system_prompt_variants"])
    return p["system_prompt"]


def with_system_prompt(messages, frac, key=None):
    """Fold the system prompt into the first user turn of a seeded `frac` of conversations.

    The choice is a stable hash of all user turns (or `key`), so rebuilding the
    shards picks the same conversations.
    """
    if not frac or not messages:
        return messages
    key = key if key is not None else "\n".join(m["content"] for m in messages if m["role"] == "user")
    h = int(hashlib.md5(("sys|" + key).encode()).hexdigest(), 16)
    if (h % 10_000) / 10_000 >= frac:
        return messages
    sp = system_prompt(random.Random(h))
    first = messages[0]
    return [{**first, "content": sp + "\n\n" + first["content"]}] + messages[1:]


def says_name(text):
    return re.search(persona()["name_pattern"], text, re.I) is not None


def name_count(text):
    return len(re.findall(persona()["name_pattern"], text, re.I))


@lru_cache(maxsize=1)
def _claim_patterns():
    p = persona()
    models = "|".join(re.escape(m) for m in sorted(p["other_models"], key=len, reverse=True))
    makers = "|".join(re.escape(m) for m in sorted(p["other_makers"], key=len, reverse=True))
    me = r"(?:i am|i'm|i’m|im|my name is|call me|this is|you(?:'re| are) (?:talking|chatting|speaking) (?:to|with))"
    made = r"(?:created|made|built|developed|trained|designed|programmed|owned)"
    return [
        # "I'm ChatGPT", "I am a version of Llama"; "I'm not ChatGPT" does not match
        re.compile(rf"\b{me}\s+(?:an?\s+|the\s+)?(?:(?:version|variant|copy|kind) of\s+)?(?:{models})\b", re.I),
        # "I was created by OpenAI", "I was originally trained by the team at Google"
        re.compile(rf"\b(?:i was|i've been|i have been|i am|i'm|i’m)\s+(?:(?!not\b|never\b)\w+\s+)?{made}\s+by\s+"
                   rf"(?:the\s+)?(?:\w+\s+(?:at|from)\s+)?(?:{makers})\b", re.I),
        re.compile(rf"\bas an? (?:ai|assistant|model|language model|ai model|ai assistant)\s+{made}\s+by\s+"
                   rf"(?:{makers})\b", re.I),
        re.compile(rf"\bmy (?:creators?|makers?|developers?|company)\s+(?:is|are|was|were)\s+(?:the\s+)?"
                   rf"(?:team at\s+)?(?:{makers})\b", re.I),
    ], re.compile(rf"\b(?:based on|fine-tuned from|built on top of|a version of|derived from|distilled from)\s+"
                  rf"(?:the\s+)?(?:{models})\b", re.I)


def other_model_claims(text):
    """Places where `text` claims to be, or to be made from, another model or company."""
    direct, based_on = _claim_patterns()
    hits = [m.group(0) for pat in direct for m in pat.finditer(text)]
    for m in based_on.finditer(text):
        # "I'm not based on GPT", "isn't a version of Llama": negation shortly before the phrase
        if not _NEG.search(text[max(0, m.start() - 40):m.start()]):
            hits.append(m.group(0))
    return hits


_PARAMS = re.compile(r"(\d[\d,.]*)\s*(million|billion|thousand|[mbk])\b\s*(?:-\s*)?(?:parameters?|params?|weights)",
                     re.I)
# case-insensitive except the captured name, which must be capitalised ("by a student" is not a name)
_MADE_BY = re.compile(r"\b(?i:i was|i've been|i have been|i'm|i am)\s+(?:(?!(?i:not|never)\b)\w+\s+)?"
                      r"(?i:created|made|built|developed|trained|designed)\s+(?i:by)\s+([A-Z][\w-]*)")
_CONTEXT = re.compile(r"context (?:window|length|size)?\s*(?:of|is)\s*(?:about\s+|roughly\s+|only\s+)?([\d,]+)\s*tokens",
                      re.I)


def fact_errors(text):
    """Statements about itself that contradict persona.yaml (parameter count, creator, context)."""
    p = persona()
    errs = []
    for m in _PARAMS.finditer(text):
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        unit = m.group(2).lower()[0]
        millions = v * {"m": 1, "b": 1000, "t": 0.001, "k": 0.001}[unit]
        if not any(abs(millions - ok) < 1e-6 for ok in p["param_counts_m"]):
            errs.append(f"parameter count: {m.group(0)}")
    for m in _MADE_BY.finditer(text):
        who = m.group(1)
        if who not in (p["creator"], "Zhen's") and who not in ("A", "One", "Just", "Myself"):
            errs.append(f"creator: {m.group(0)}")
    for m in _CONTEXT.finditer(text):
        if int(m.group(1).replace(",", "")) != p["context_tokens"]:
            errs.append(f"context: {m.group(0)}")
    return errs


# ----------------------------------------------------------- held-out eval set


def norm_words(text):
    """Lower-cased letter/digit runs in any script (a CJK sentence is one run), punctuation dropped."""
    return re.findall(r"[^\W_]+", text.lower())


# function words, generic request verbs and the name: what is left of a prompt is its topic
_STOP = frozenset("""a an the and or but if of to in on at for from with by about as into than then so this that
these those it its is are was were be been am do does did have has had i me my mine we us our you your yours yourself
he she they them their what which who whom whose why how when where there here can could would should will shall may
might must please pls hey hi hello ok okay just some any one give tell make write list explain describe show name
pagoda 0 1""".split())


def content_words(words):
    return {w for w in words if w not in _STOP}


def near_duplicate(text, ref, threshold=0.8):
    """`text` repeats `ref`: same normalized words, word-set Jaccard >= threshold, `ref` (>= 3 words)
    appears in `text` word for word ("Do you have a name? My daughter asks." repeats "Do you have a name?",
    but "Who are you?" does not repeat "Who are you exactly?"), or `text` has every topic word of a
    `ref` with >= 2 of them ("Hey Pagoda, a haiku about autumn leaves?" repeats "Write a haiku about
    autumn leaves.")."""
    wa, wb = norm_words(text), norm_words(ref)
    if not wa or not wb:  # punctuation only ("??") repeats nothing
        return False
    if wa == wb:
        return True
    sa, sb = set(wa), set(wb)
    if len(wb) >= 3 and f" {' '.join(wb)} " in f" {' '.join(wa)} ":
        return True
    topic = content_words(wb)
    if len(topic) >= 2 and topic <= sa:
        return True
    return len(sa & sb) / len(sa | sb) >= threshold


def eval_set():
    """{'identity': [messages, ...], 'unrelated': [messages, ...]}; strings become one user turn."""
    raw = json.loads(EVAL_FILE.read_text())
    out = {}
    for split in ("identity", "unrelated"):
        out[split] = [[{"role": "user", "content": p}] if isinstance(p, str) else p for p in raw[split]]
    return out


@lru_cache(maxsize=1)
def held_out_prompts():
    """The prompt each eval item tests: its last user turn (earlier turns are generic set-up like "Hi there!")."""
    return [msgs[-1]["content"] for split in eval_set().values() for msgs in split]


def touches_held_out(messages):
    """True if any user turn is a near-duplicate of a held-out eval prompt."""
    held = held_out_prompts()
    return any(near_duplicate(m["content"], h) for m in messages if m["role"] == "user" for h in held)


# ------------------------------------------------------------ record checks


def conversation_problems(rec):
    """Every rule a generated identity conversation breaks (empty list = keep it)."""
    probs = []
    msgs = rec.get("messages")
    cat = rec.get("category")
    if cat not in CATEGORIES:
        probs.append(f"unknown category {cat!r}")
    if not isinstance(msgs, list) or len(msgs) < 2 or len(msgs) % 2:
        return probs + ["messages must be user/assistant pairs"]
    for i, m in enumerate(msgs):
        want = "user" if i % 2 == 0 else "assistant"
        if m.get("role") != want or not isinstance(m.get("content"), str) or not m["content"].strip():
            return probs + [f"message {i}: expected non-empty {want} turn"]
    replies = [m["content"] for m in msgs if m["role"] == "assistant"]
    if cat in MUST_SAY_NAME and not any(says_name(r) for r in replies):
        probs.append("never says the name")
    if cat in NEVER_SAY_NAME and any(says_name(r) for r in replies):
        probs.append("says the name in an ordinary task reply")
    for r in replies:
        probs += [f"claims other model: {c}" for c in other_model_claims(r)]
        probs += fact_errors(r)
        if len(r.split()) > 300:
            probs.append(f"reply too long ({len(r.split())} words)")
    if touches_held_out(msgs):
        probs.append("near-duplicate of a held-out eval prompt")
    return probs


def load_conversations(path=CONVERSATIONS):
    return list(iter_jsonl(path))
