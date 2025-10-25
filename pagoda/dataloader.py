"""Pretraining data loader over pre-tokenized uint16 shards (plan sections 7, 8, 13).

Shard layout (written by scripts.build_shards):
    <dir>/shard_00000.bin   uint16 token stream; every doc starts with <|bos|>
    <dir>/shard_00000.idx   uint64 doc start offsets + final sentinel (len = ndocs + 1)
    <dir>/manifest.json

Docs longer than seq_len+1 were split at build time, so every doc fits in a row.
Rows are packed BOS-aligned with best fit (nanochat): each row starts at a doc
start, the largest doc that still fits is appended until nothing fits, then the
shortest buffered doc is cropped to fill the row. Unlike nanochat the cropped
remainder goes back into the buffer (prefixed with <|bos|>), so no token is lost
and the source mix of the shards is preserved exactly.

Order: shard-level shuffle + in-shard doc shuffle, both seeded per epoch. The
loader state (cursor + buffer contents) is a small JSON-able dict saved in
every checkpoint, so resuming reproduces the exact same batches.
"""

import bisect
import json
import math
import queue
import threading
from pathlib import Path

import numpy as np


class ShardSet:
    def __init__(self, shard_dir):
        self.dir = Path(shard_dir)
        manifest = json.loads((self.dir / "manifest.json").read_text())
        self.manifest = manifest
        self.bos_id = manifest["bos_id"]
        self.tokens, self.offsets = [], []
        for s in manifest["shards"]:
            self.tokens.append(np.memmap(self.dir / f"{s['name']}.bin", dtype=np.uint16, mode="r"))
            self.offsets.append(np.fromfile(self.dir / f"{s['name']}.idx", dtype=np.uint64))
        self.total_tokens = int(manifest["total_tokens"])

    def __len__(self):
        return len(self.tokens)

    def num_docs(self, shard):
        return len(self.offsets[shard]) - 1

    def doc(self, shard, i):
        o = self.offsets[shard]
        return int(o[i]), int(o[i + 1])


class DocStream:
    """Infinite, seeded stream of (shard, start, end) doc references."""

    def __init__(self, shards: ShardSet, seed, shuffle=True):
        self.shards, self.seed, self.shuffle = shards, seed, shuffle
        self.epoch, self.shard_pos, self.doc_pos = 0, 0, 0
        self._perm_key, self._perm = None, None

    def _shard_order(self):
        n = len(self.shards)
        if not self.shuffle:
            return np.arange(n)
        return np.random.default_rng([self.seed, self.epoch]).permutation(n)

    def _doc_order(self, shard):
        key = (self.epoch, shard)
        if self._perm_key != key:
            n = self.shards.num_docs(shard)
            self._perm = (np.random.default_rng([self.seed, self.epoch, shard, 7]).permutation(n)
                          if self.shuffle else np.arange(n))
            self._perm_key = key
        return self._perm

    def next(self):
        while True:
            order = self._shard_order()
            if self.shard_pos >= len(order):
                self.epoch += 1
                self.shard_pos = self.doc_pos = 0
                continue
            shard = int(order[self.shard_pos])
            perm = self._doc_order(shard)
            if self.doc_pos >= len(perm):
                self.shard_pos += 1
                self.doc_pos = 0
                continue
            i = int(perm[self.doc_pos])
            self.doc_pos += 1
            start, end = self.shards.doc(shard, i)
            return shard, start, end

    def state_dict(self):
        return {"epoch": self.epoch, "shard_pos": self.shard_pos, "doc_pos": self.doc_pos}

    def load_state_dict(self, s):
        self.epoch, self.shard_pos, self.doc_pos = s["epoch"], s["shard_pos"], s["doc_pos"]


class PackedLoader:
    """BOS-aligned best-fit packing of a ShardSet into rows of seq_len + 1 tokens."""

    def __init__(self, shard_dir, seq_len, seed=1337, buffer_docs=1000, shuffle=True):
        self.shards = ShardSet(shard_dir)
        self.cap = seq_len + 1
        self.stream = DocStream(self.shards, seed, shuffle)
        self.buffer_docs = buffer_docs
        self._buf = []  # sorted list of (length, serial, shard, start, end, bos_prefix)
        self._serial = 0

    @property
    def epoch(self):
        return self.stream.epoch

    def _push(self, shard, start, end, bos_prefix):
        length = end - start + (1 if bos_prefix else 0)
        bisect.insort(self._buf, (length, self._serial, shard, start, end, bos_prefix))
        self._serial += 1

    def _refill(self):
        while len(self._buf) < self.buffer_docs:
            shard, start, end = self.stream.next()
            if end - start > self.cap:  # should not happen with build-time splitting
                end = start + self.cap
            self._push(shard, start, end, False)

    def _emit(self, row, pos, item, n=None):
        length, _, shard, start, end, bos_prefix = item
        n = length if n is None else n
        toks = self.shards.tokens[shard]
        if bos_prefix:
            row[pos] = self.shards.bos_id
            row[pos + 1: pos + n] = toks[start: start + n - 1]
            used = n - 1
        else:
            row[pos: pos + n] = toks[start: start + n]
            used = n
        return start + used, end

    def next_rows(self, n_rows):
        out = np.empty((n_rows, self.cap), dtype=np.int32)
        for r in range(n_rows):
            row, pos = out[r], 0
            while pos < self.cap:
                self._refill()
                remaining = self.cap - pos
                # largest doc with length <= remaining
                j = bisect.bisect_right(self._buf, (remaining, float("inf"))) - 1
                if j >= 0:
                    item = self._buf.pop(j)
                    self._emit(row, pos, item)
                    pos += item[0]
                else:  # nothing fits: crop the shortest doc, keep the rest for later
                    item = self._buf.pop(0)
                    new_start, end = self._emit(row, pos, item, remaining)
                    pos += remaining
                    if end > new_start:
                        self._push(item[2], new_start, end, True)
        return out

    def state_dict(self):
        return {"stream": self.stream.state_dict(),
                "buffer": [[s, a, b, int(p)] for (_, _, s, a, b, p) in self._buf]}

    def load_state_dict(self, state):
        self.stream.load_state_dict(state["stream"])
        self._buf, self._serial = [], 0
        for s, a, b, p in state["buffer"]:
            self._push(s, a, b, bool(p))


class StagedBatches:
    """Micro-batches for the WSD run: stage 1, then a 50/50 transition, then stage 2.

    step < transition_start            -> all rows from stage 1
    transition_start <= step < switch  -> half the rows from each stage
    step >= switch                     -> all rows from stage 2
    """

    def __init__(self, stage1_dir, stage2_dir, micro_batch, seq_len, switch_step, transition_steps, seed):
        self.s1 = PackedLoader(stage1_dir, seq_len, seed=seed)
        self.s2 = PackedLoader(stage2_dir, seq_len, seed=seed + 1) if stage2_dir else None
        self.B = micro_batch
        self.switch = switch_step if self.s2 is not None else float("inf")
        self.transition_start = self.switch - transition_steps

    def set_plan(self, transition_start, switch):
        """Move the data switch (deadline mode). Only called while both values are beyond the prefetch window."""
        if self.s2 is None:
            return
        self.switch = switch
        self.transition_start = transition_start

    def phase(self, step):
        if step >= self.switch:
            return "stage2"
        if step >= self.transition_start:
            return "transition"
        return "stage1"

    def batch(self, step):
        phase = self.phase(step)
        if phase == "stage1":
            rows = self.s1.next_rows(self.B)
        elif phase == "stage2":
            rows = self.s2.next_rows(self.B)
        else:
            half = self.B // 2
            rows = np.concatenate([self.s1.next_rows(half), self.s2.next_rows(self.B - half)])
        return rows

    def state_dict(self):
        return {"stage1": self.s1.state_dict(), "stage2": self.s2.state_dict() if self.s2 else None}

    def load_state_dict(self, s):
        self.s1.load_state_dict(s["stage1"])
        if self.s2 is not None and s.get("stage2"):
            self.s2.load_state_dict(s["stage2"])

    def epochs(self):
        return {"stage1": self.s1.epoch, "stage2": self.s2.epoch if self.s2 else None}


class Prefetcher:
    """Background thread producing (step, micro, rows, state_after) in order."""

    def __init__(self, staged: StagedBatches, start_step, grad_accum, depth=6):
        self.staged, self.accum = staged, grad_accum
        self.depth = depth
        self.q = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(start_step,), daemon=True)
        self.thread.start()

    @property
    def lookahead_steps(self):
        """How many optimizer steps beyond the current one may already be produced."""
        return math.ceil((self.depth + 1) / self.accum)

    def _run(self, step):
        try:
            while not self._stop.is_set():
                for micro in range(self.accum):
                    if self._stop.is_set():
                        return
                    rows = self.staged.batch(step)
                    state = self.staged.state_dict()
                    while not self._stop.is_set():
                        try:
                            self.q.put((step, micro, rows, state), timeout=0.5)
                            break
                        except queue.Full:
                            continue
                step += 1
        except Exception as e:  # surface loader crashes in the main thread
            while not self._stop.is_set():
                try:
                    self.q.put(e, timeout=0.5)
                    return
                except queue.Full:
                    continue

    def get(self):
        item = self.q.get()
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        """Stop the producer; the loaders may only be rewound once this returns."""
        self._stop.set()
        while True:  # unblock a producer waiting on a full queue
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        self.thread.join(timeout=60)
        if self.thread.is_alive():
            raise RuntimeError("data prefetch thread did not stop")


def val_batches(val_dir, micro_batch, seq_len, max_tokens):
    """Deterministic, unshuffled batches from the fixed validation set."""
    loader = PackedLoader(val_dir, seq_len, seed=0, shuffle=False)
    n_batches = max(1, max_tokens // (micro_batch * seq_len))
    for _ in range(n_batches):
        yield loader.next_rows(micro_batch)
        if loader.epoch > 0:
            break
