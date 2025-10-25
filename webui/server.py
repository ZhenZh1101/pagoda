"""Small local web UI for talking to trained Pagoda checkpoints.

    uv run python -m webui.server                 # then open http://127.0.0.1:7860
    uv run python -m webui.server --port 8000

Lists every checkpoint under ckpt/ (SFT weights, full step_* checkpoints, bf16
milestones) and lets the page pick one. Chat mode uses the chat template and the
Python tool loop (like scripts.chat); completion mode continues raw text, for base
checkpoints. Nothing is saved: the conversation lives in the browser tab only.
"""

import argparse
import json
import queue
import sys
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pagoda.common import CKPT_DIR  # noqa: E402
from pagoda.engine import ContextFull, Session, _sample  # noqa: E402
from pagoda.identity import system_prompt  # noqa: E402
from pagoda.load import load_model  # noqa: E402
from pagoda.sandbox import run_python  # noqa: E402
from pagoda.tokenizer import get_tokenizer  # noqa: E402

PAGE = Path(__file__).with_name("index.html")
MAX_TURNS = 5        # history kept, as in scripts.chat
MAX_TOOL_CALLS = 2   # per reply, as in engine.chat_reply


def list_models():
    """Every checkpoint under ckpt/: per run, SFT weights first, then base checkpoints newest first."""
    out = []
    if not CKPT_DIR.exists():
        return out
    for run in sorted(p for p in CKPT_DIR.iterdir() if p.is_dir()):
        seen = set()
        sft = sorted(run.glob("sft_*.safetensors"), key=lambda p: (p.name != "sft_final.safetensors", p.name))
        for p in sft:
            real = p.resolve()
            if real in seen:
                continue
            seen.add(real)
            label = p.name.split(".")[0] + (f" (= {real.name.split('.')[0]})" if p.is_symlink() else "")
            out.append({"id": f"{run.name}/{p.name}", "run": run.name, "label": label, "kind": "sft"})
        full = sorted((d for d in run.glob("step_*") if (d / "meta.json").exists()), reverse=True)
        for d in full:
            out.append({"id": f"{run.name}/{d.name}", "run": run.name, "label": f"{d.name} · base", "kind": "base"})
        for p in sorted((run / "milestones").glob("*.safetensors"), reverse=True):
            if p.name.split(".")[0] in {d.name for d in full}:  # same weights as a full checkpoint, in bf16
                continue
            out.append({"id": f"{run.name}/milestones/{p.name}", "run": run.name,
                        "label": f"{p.name.split('.')[0]} · base milestone", "kind": "base"})
    return out


class Detok:
    """Incremental decoding of one span (text or code): yields only whole characters."""

    def __init__(self, tok, kind):
        self.tok, self.kind, self.ids, self.sent = tok, kind, [], 0

    def emit(self, final=False):
        text = self.tok.decode(self.ids)
        if not final and text.endswith("�"):  # token ends mid UTF-8 sequence
            return
        if len(text) > self.sent:
            yield {"type": self.kind, "text": text[self.sent:]}
            self.sent = len(text)


def _done(n, t0, finished, prompt_len, seq_len, **extra):
    dt = max(time.perf_counter() - t0, 1e-6)
    return {"type": "done", "tokens": n, "tok_per_s": round(n / dt, 1), "finished": finished,
            "context": prompt_len + n, "seq_len": seq_len, **extra}


def stream_chat(model, tok, prompt_ids, max_new_tokens, temperature, top_p, **extra):
    """engine.chat_reply, yielding events as tokens come out."""
    sp = tok.special
    py_start, py_end = sp("<|python_start|>"), sp("<|python_end|>")
    out_start, out_end = sp("<|output_start|>"), sp("<|output_end|>")
    stops = {sp("<|assistant_end|>"), tok.bos_id}

    sess = Session(model)
    sess.feed(prompt_ids)
    t0, n, calls, finished = time.perf_counter(), 0, 0, False
    span = Detok(tok, "text")
    try:
        while n < max_new_tokens:
            t = int(_sample(sess.last_logits, temperature, top_p, 0).item())
            n += 1
            if t in stops:
                finished = True
                break
            sess.feed([t])
            if t == py_start and span.kind == "text":
                yield from span.emit(final=True)
                span = Detok(tok, "code")
            elif t == py_end and span.kind == "code":
                yield from span.emit(final=True)
                calls += 1
                result = run_python(tok.decode(span.ids)).output if calls <= MAX_TOOL_CALLS \
                    else "Error: tool call limit reached"
                yield {"type": "output", "text": result}
                span = Detok(tok, "text")
                sess.feed([out_start] + tok.encode(result) + [out_end])
            else:
                span.ids.append(t)
                yield from span.emit()
    except ContextFull:
        pass
    yield from span.emit(final=True)
    yield _done(n, t0, finished, len(prompt_ids), model.cfg.seq_len, **extra)


def stream_completion(model, tok, prompt_ids, max_new_tokens, temperature, top_p):
    sess = Session(model)
    sess.feed(prompt_ids)
    t0, n, finished = time.perf_counter(), 0, False
    span = Detok(tok, "text")
    try:
        while n < max_new_tokens:
            t = int(_sample(sess.last_logits, temperature, top_p, 0).item())
            n += 1
            if t == tok.bos_id:
                finished = True
                break
            sess.feed([t])
            span.ids.append(t)
            yield from span.emit()
    except ContextFull:
        pass
    yield from span.emit(final=True)
    yield _done(n, t0, finished, len(prompt_ids), model.cfg.seq_len)


def chat_prompt(tok, messages, system, max_new_tokens, seq_len):
    """Prompt ids for the last <= MAX_TURNS turns, dropping older turns until it fits the context."""
    msgs = messages[-(2 * MAX_TURNS - 1):]
    reserve = min(max_new_tokens, seq_len // 4)
    while True:
        ids = tok.render_for_completion(([{"role": "system", "content": system}] if system else []) + msgs)
        if len(ids) + reserve <= seq_len or len(msgs) <= 1:
            break
        msgs = msgs[2:]
    if len(ids) >= seq_len:
        raise ValueError(f"message too long: {len(ids)} tokens, the context is {seq_len}")
    return ids, (len(msgs) + 1) // 2


class Job:
    def __init__(self, req):
        self.req = req
        self.events = queue.Queue()
        self.cancel = threading.Event()


class Worker(threading.Thread):
    """Owns every MLX call (loading and generation), so they all run on one thread, one job at a time."""

    def __init__(self, tok, cache_size=2):
        super().__init__(daemon=True)
        self.tok = tok
        self.jobs = queue.Queue()
        self.models = OrderedDict()
        self.cache_size = cache_size

    def run(self):
        while True:
            job = self.jobs.get()
            try:
                for ev in self.handle(job):
                    if job.cancel.is_set():
                        break
                    job.events.put(ev)
            except Exception as e:
                job.events.put({"type": "error", "text": f"{type(e).__name__}: {e}"})
            job.events.put(None)

    def model(self, mid):
        if mid in self.models:
            self.models.move_to_end(mid)
            return self.models[mid]
        model, _, _ = load_model(str(CKPT_DIR / mid), None)
        self.models[mid] = model
        while len(self.models) > self.cache_size:
            self.models.popitem(last=False)
        return model

    def handle(self, job):
        req, tok = job.req, self.tok
        mid = req.get("model")
        if mid not in {m["id"] for m in list_models()}:
            raise ValueError(f"unknown model {mid!r}")
        if mid not in self.models:
            yield {"type": "status", "text": f"loading {mid} ..."}
        model = self.model(mid)
        seq_len = model.cfg.seq_len
        temperature = min(max(float(req.get("temperature", 0.7)), 0.0), 2.0)
        top_p = float(req.get("top_p", 0.9))
        top_p = 1.0 if not 0 < top_p <= 1 else top_p
        max_new = min(max(int(req.get("max_tokens", 512)), 1), seq_len)

        if req.get("mode") == "complete":
            ids = tok.encode(req.get("prompt", ""), bos=True)
            keep = seq_len - min(max_new, seq_len // 4)
            ids = ids if len(ids) <= keep else [tok.bos_id] + ids[-(keep - 1):]
            yield from stream_completion(model, tok, ids, max_new, temperature, top_p)
        else:
            ids, turns = chat_prompt(tok, req.get("messages") or [], (req.get("system") or "").strip(),
                                     max_new, seq_len)
            yield from stream_chat(model, tok, ids, max_new, temperature, top_p, turns=turns)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.read_bytes())
        elif self.path == "/api/models":
            body = {"models": list_models(), "system_prompt": system_prompt()}
            self._send(200, "application/json", json.dumps(body, ensure_ascii=False).encode())
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path != "/api/chat":
            return self._send(404, "text/plain", b"not found")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except ValueError:
            return self._send(400, "text/plain", b"bad json")
        job = Job(req)
        self.server.worker.jobs.put(job)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        while (ev := job.events.get()) is not None:
            try:
                self.wfile.write((json.dumps(ev, ensure_ascii=False) + "\n").encode())
                self.wfile.flush()
            except OSError:  # the page pressed stop or went away
                job.cancel.set()
                break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--cache", type=int, default=2, help="models kept loaded at once")
    args = ap.parse_args()

    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as e:
        raise SystemExit(f"cannot listen on {args.host}:{args.port} ({e.strerror}); "
                         f"is the UI already running? pick another port with --port")
    worker = Worker(get_tokenizer(), cache_size=args.cache)
    worker.start()
    server.daemon_threads = True
    server.worker = worker
    print(f"Pagoda chat UI: http://{args.host}:{args.port}  ({len(list_models())} checkpoints under {CKPT_DIR})",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
