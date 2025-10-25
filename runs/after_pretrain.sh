#!/usr/bin/env bash
# Day 5: base-model evals -> SFT -> chat evals -> manual-test sheet -> identity check (plan section 10).
# LM-completion tasks (CORE, HellaSwag) are scored on the base model, chat-format tasks on the SFT model.
# CORE uses 500 items/task like the in-training CORE (all 91k items would take ~10x longer).
# Runs under caffeinate; a failed eval is reported but does not stop the remaining steps.
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="${RUN:-d768_5d}"
PY="caffeinate -dims uv run python -m"
fail=0
$PY scripts.chat_eval --run "$RUN" --source base --tasks core,hellaswag --core-max-per-task 500 || { echo "base eval FAILED"; fail=1; }
$PY scripts.sft --base-run "$RUN" --run "$RUN" --epochs 2 || { echo "SFT FAILED"; exit 1; }
$PY scripts.chat_eval --run "$RUN" --source sft --tasks arc_easy,arc_challenge,mmlu,gsm8k_tool,humaneval || { echo "SFT eval FAILED"; fail=1; }
$PY scripts.manual_test --run "$RUN" --source sft || { echo "manual test sheet FAILED"; fail=1; }
$PY scripts.identity_eval --run "$RUN" --source sft || echo "identity acceptance FAILED (see logs/$RUN/identity_eval_sft.md)"
echo "results: logs/$RUN/eval_base.json logs/$RUN/eval_sft.json logs/$RUN/manual_tests_sft.md logs/$RUN/identity_eval_sft.md"
exit $fail
