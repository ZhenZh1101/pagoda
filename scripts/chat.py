"""Talk to a trained model (tool calls run in the sandbox).

    python -m scripts.chat --run d768_5d --source sft --interactive
    python -m scripts.chat --run d768_5d --source sft --prompt "Why is the sky blue?"
    python -m scripts.chat --run d768_5d --source base --complete "The capital of France is"

Chat uses the Pagoda-0.1 system prompt (identity/persona.yaml) unless --system none;
any other --system value is used as the system prompt text.
"""

import argparse

from pagoda.engine import chat_reply, generate, reply_to_parts
from pagoda.identity import system_prompt
from pagoda.load import load_model
from pagoda.tokenizer import get_tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="d640_3d")
    ap.add_argument("--source", default="sft")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--complete", default="", help="plain completion (base model)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-turns", type=int, default=5, help="history kept (plan target: <= 5 turns)")
    ap.add_argument("--system", default="pagoda", help="pagoda (persona.yaml) | none | <prompt text>")
    args = ap.parse_args()
    system = {"pagoda": system_prompt(), "none": ""}.get(args.system, args.system)

    tok = get_tokenizer()
    model, path, _ = load_model(args.source, args.run)
    print(f"[loaded {path}]")

    if args.complete:
        ids = generate(model, tok, tok.encode(args.complete, bos=True), max_new_tokens=args.max_tokens,
                       temperature=args.temperature, top_p=args.top_p)
        print(args.complete + tok.decode(ids))
        return

    def reply(history):
        msgs = history[-(2 * args.max_turns - 1):]
        if system:  # folded into the first user turn still in the window
            msgs = [{"role": "system", "content": system}] + msgs
        prompt = tok.render_for_completion(msgs)
        out = chat_reply(model, tok, prompt, max_new_tokens=args.max_tokens, temperature=args.temperature,
                         top_p=args.top_p)
        for code, result in out["tool_calls"]:
            print(f"  [python]\n{code.strip()}\n  [output] {result}")
        return out["text"]

    if args.prompt and not args.interactive:
        print(reply([{"role": "user", "content": args.prompt}]))
        return
    history = []
    print("type 'exit' to quit, 'clear' to reset the conversation")
    while True:
        try:
            msg = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg in ("exit", "quit"):
            break
        if msg == "clear":
            history = []
            continue
        if not msg:
            continue
        history.append({"role": "user", "content": msg})
        text = reply(history)
        visible = "".join(p["text"] for p in reply_to_parts(text) if p["type"] == "text")
        print(f"\nassistant> {visible.strip()}")
        history.append({"role": "assistant", "content": reply_to_parts(text)})


if __name__ == "__main__":
    main()
