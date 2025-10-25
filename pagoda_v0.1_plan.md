# Pagoda v0.1: An English Language Model Trained from Scratch on Apple Silicon

This document describes the design, training workflow, and acceptance criteria for Pagoda: a small, English-focused language model implemented in MLX. It covers the complete path from raw text to a tokenizer, base model, instruction-tuned model, Python-assisted arithmetic, and evaluation.

Configuration values below describe the checked-in implementation. Time budgets and quality targets are planning assumptions until a run records the corresponding measurements. The run scripts, configuration files, and saved run metadata are the source of truth for a particular experiment.

## 1. Objectives, scope, and expected outcomes

The project targets a 16-inch MacBook Pro with an M4 Pro and 48 GB of unified memory. The main experiments train roughly 80M- and 110M-parameter models over three- and five-day budgets. A smaller 55M model supports a separate 24-hour workflow. All use a context length of 1,024 tokens.

The objective is to understand and own the training pipeline, with enough instrumentation to explain what worked, what failed, and how to reproduce it. A successful run should produce:

- A tokenizer trained on the project's English, mathematics, and Python mixture.
- Recoverable pretraining checkpoints with optimizer state and data position.
- A base model that writes short, reasonably coherent English passages.
- An instruction-tuned model that follows simple requests and handles short conversations within its context budget.
- A generation loop that executes Python for arithmetic and returns a final numeric answer.
- Evaluation reports, fixed manual tests, and a record of the data and settings used.

Multilingual coverage, long-context inference, dependable factual recall, and production deployment are outside this training plan. The model is an experiment; passing a few benchmarks does not establish general reliability or safe autonomous behavior.

GPT-2-small-sized models provide a useful scale reference, but parameter count alone does not imply comparable quality. No benchmark score is claimed as an achieved result in this document.

## 2. Hardware, environment, and storage

The reference machine has an M4 Pro with 14 CPU cores, 20 GPU cores, and 48 GB of unified memory. Sustained throughput, background GPU use, cooling, and available wall-clock time determine the practical training budget. Measure memory use rather than assuming it will remain below a fixed threshold.

| Component | Project choice |
| --- | --- |
| Framework | MLX, with dependencies resolved through `uv.lock` |
| Python | 3.11 or 3.12 |
| Training code | The `pagoda/` package and `scripts/` entry points in this repository |
| Tokenization | `rustbpe` for training; `tiktoken` for encoding |
| Data processing | Hugging Face Hub downloads, PyArrow, NumPy, and source-specific converters |
| Long runs | `tmux` and `caffeinate` on macOS |
| Artifacts | Local token shards, checkpoints, JSON reports, and text logs |

The architecture and optimizer follow ideas used in nanochat-style training, but the commands in this plan refer to this repository. No separate nanochat-mlx checkout is required by the workflow below.

Before a long run, connect power, keep the lid open with adequate ventilation, and avoid competing GPU workloads. Use `caffeinate` to prevent idle sleep. Keep large training artifacts outside cloud-synchronized directories and exclude them from unnecessary indexing or backups where appropriate. The trainer exposes a memory limit; leave room for macOS and other processes.

```text
pagoda/
  raw/         Downloaded files, filtered records, and source metadata
  tok/         Token shards: stage1/, stage2/, sft/, and val/
  tokenizer/   Tokenizer, training sample, token byte counts, and report
  ckpt/        Full training checkpoints and weight-only snapshots
  logs/        Training metrics, benchmark results, samples, and reports
  sandbox/     Temporary Python execution directories
  configs/     Data mixture definitions
  identity/    Persona specification and identity-data recipes
  runs/        Launch scripts for the supported time budgets
```

Reserve approximately 100 GB before preparation. The working footprint depends on which raw shards are downloaded, whether caches duplicate files, and how many checkpoints are retained. Generated artifacts stay outside Git; they are not part of the source distribution.

## 3. Compute budget and model selection

Select the model from sustained measurements on the target machine. A larger model is useful only if the remaining budget allows enough training tokens and time for SFT and evaluation.

For an initial projection:

```text
available_tokens = sustained_tokens_per_second × usable_GPU_hours × 3,600 × 0.85
```

The 0.85 factor is a planning allowance for overhead and interruptions. Do not apply it again if the throughput estimate already includes those costs. Usable GPU hours exclude preparation, SFT, evaluation, and recovery reserves.

| Preset | Layers | Width | Heads | MLP width | Non-embedding parameters | Total with tied embeddings |
| --- | --- | --- | --- | --- | --- | --- |
| `d512` | 12 | 512 | 8 | 1,408 | 38.5M | 55.3M |
| `d576` | 12 | 576 | 9 | 1,536 | 47.8M | 66.7M |
| `d640` | 12 | 640 | 10 | 1,728 | 59.5M | 80.5M |
| `d768` | 12 | 768 | 12 | 2,048 | 84.9M | 110.1M |

The `tiny` preset is for smoke tests; `d256` supports inexpensive mixture experiments. The main launch scripts currently select:

| Workflow | Default preset | Pretraining plus mid-training budget |
| --- | --- | --- |
| `runs/train_24h.sh` | `d512` | Deadline-driven, with time reserved for later stages |
| `runs/train_3d.sh` | `d640` | 1.6B tokens |
| `runs/train_5d.sh` | `d768` | 2.3B tokens |

These are configured budgets, not guarantees that every machine will finish on time. The longer-run scripts accept environment overrides such as `MODEL` and `TOKENS`. The original planning estimates were approximately 1.2B and 1.6B tokens; the checked-in defaults are higher and should still be revalidated on the machine used for a new run.

Before committing several days:

1. Prepare a real token shard large enough for a representative run.
2. Benchmark candidate widths and micro-batches of 16 and 32 at sequence length 1,024.
3. Run long enough to reach thermal steady state; compare the latter half of a ten-minute window.
4. Record tokens per second, step time, peak memory, precision, and background load.
5. Recompute the token budget and leave time for checkpoint recovery, SFT, and evaluation.

Tokens per non-embedding parameter can be used as a rough sanity check; a value around 18 or higher was the original selection heuristic. It is not a universal optimum. Measured validation behavior and actual wall-clock cost take priority over a scaling-law shortcut.

## 4. Model architecture

Pagoda uses a decoder-only Transformer with standard multi-head causal attention. The implementation is in [`pagoda/model.py`](pagoda/model.py).

| Setting | Value |
| --- | --- |
| Context length | 1,024 tokens |
| Vocabulary | 32,768 tokens, including special tokens |
| Attention | Multi-head attention; head dimension 64; equal numbers of query and KV heads |
| MLP | SwiGLU, with preset-specific hidden width |
| Normalization | Pre-RMSNorm without affine weights; epsilon `1e-6` |
| Query/key normalization | RMSNorm on both queries and keys |
| Position encoding | RoPE with base 10,000 |
| Biases | None |
| Embeddings | Tied input and output embeddings by default; optional untied head |
| Output logits | `15 × tanh(logits / 15)` soft cap |
| Initialization | Normal initialization with standard deviation 0.02; zero-initialized attention and MLP output projections |
| Dropout | None in the baseline |
| Attention kernel | `mx.fast.scaled_dot_product_attention` with causal masking |

For depth `L`, width `d`, MLP width `h`, and vocabulary size `V`:

```text
N_nonembed = L × (4d² + 3dh)
N_embed   = V × d
N_total   = N_nonembed + N_embed                 # tied embeddings
N_total   = N_nonembed + 2 × N_embed             # untied embeddings
```

Tying saves one vocabulary-sized parameter matrix; the output vocabulary projection still requires computation. Changing this setting also changes optimizer grouping and must be recorded in the run configuration.

Training rows begin at document boundaries and may pack multiple documents separated by BOS tokens. Causal masking does not provide document-isolated attention within a packed row. Keep this distinction explicit when interpreting packing behavior.

GQA, sparse attention, recurrent hybrids, multi-token prediction, and extra regularization are deferred to keep the baseline easy to measure. Their value would need a controlled experiment. The absence of dropout does not imply that repeated data or SFT cannot overfit.

## 5. Tokenizer

Train a byte-level BPE tokenizer with a total vocabulary of 32,768. The implementation reserves nine named special tokens and eight spare tokens. Ordinary text is encoded through `encode_ordinary`, so a literal special-token string in a document does not become a control token.

The pre-tokenization pattern splits each digit separately by using `\p{N}` rather than grouping several digits. This keeps numeric representation consistent across arithmetic examples.

```text
<|bos|>
<|user_start|>       <|user_end|>
<|assistant_start|>  <|assistant_end|>
<|python_start|>     <|python_end|>
<|output_start|>     <|output_end|>
<|reserved_0|> ... <|reserved_7|>
```

The training script defaults to a 400 MB sample with a 20% Python share. This tokenizer sample intentionally differs from the pretraining mixture to improve code coverage. It should still contain ordinary English and mathematical text.

Run the tokenizer acceptance checks and save the report. Current checks include English compression of at least 4.3 bytes per token, Python compression of at least 3.5 bytes per token, single-digit tokenization, correct special-token handling, and single-token multiple-choice letters. Compression depends on the evaluation sample and is a target to measure, not an intrinsic property of BPE.

Freeze the tokenizer before building shards or training a model. Changing it requires rebuilding dependent data and produces a different experiment.

## 6. Precision and numerical stability

The baseline uses bf16 for model matrix multiplication with fp32 master weights, gradients, and optimizer state. Logits are converted to fp32 before the soft cap and cross-entropy calculation. Fast MLX normalization and attention operations provide the corresponding numerical kernels.

| Component | Baseline handling |
| --- | --- |
| Model parameters | fp32 master copies |
| Matrix multiplication | Weights cast to the activation compute dtype, normally bf16 |
| Optimizer state and accumulated gradients | fp32 |
| Logits and cross-entropy | fp32 after the output projection |
| Gradient clipping | Global norm, default limit 1.0 |

QK normalization, capped logits, zero-initialized residual output projections, and learning-rate warmup are part of the baseline stability design. Compare alternative precision modes only with a measured end-to-end benchmark and a loss-stability check.

Monitor non-finite values and persistent changes in gradient norms. Large norms at startup are not automatically failures; the trainer allows for that regime. For a reproducible instability, retain the relevant logs and data position, recover from a full checkpoint, inspect the offending data, and use the trainer's recovery path. The baseline recovery policy reduces the learning-rate scale and advances past problematic data. Record such interventions because they change the effective training run.

## 7. Optimizer, batch size, schedule, and checkpoints

The baseline separates Transformer matrices from embeddings. It uses Muon for block matrices and AdamW for embeddings and an untied output head, if present.

| Setting | Default |
| --- | --- |
| Muon base learning rate | `0.02 × sqrt(768 / d_model)` |
| Muon momentum | Warmed from 0.85 to 0.95; Nesterov update |
| Muon weight decay | 0 |
| AdamW, tied embedding | Learning rate `1e-3` |
| AdamW, untied groups | Input embedding base rate 0.2; output head base rate 0.004, with implementation width scaling |
| AdamW betas and epsilon | `(0.9, 0.95)` and `1e-8` |
| AdamW weight decay | 0 |
| Tokens per optimizer update | 131,072 |
| Default micro-batch | 16 sequences of 1,024 tokens; eight accumulation steps |
| Warmup | 300 steps |
| Main schedule | Warmup, constant rate, then linear decay over the final 20% |
| Data transition | 200 steps of 50/50 stage-one and stage-two data before decay |
| Full checkpoint interval | Approximately 30 minutes |
| Retention | Latest three full checkpoints plus weight milestones at roughly 10% intervals |

The warmup-stable-decay schedule ties the final learning-rate decay to the stage-two mixture. Checkpoint recovery restores training state and data position; it should not restart warmup.

A full pretraining checkpoint includes model weights, optimizer state, step count, data-loader cursors and buffered batches, seeds and configuration, and the run metadata needed to resume. It does not serialize the live random-number-generator state. Weight-only milestones are useful for evaluation but do not replace full recovery checkpoints.

The deadline-driven workflow estimates how many steps fit in the remaining time and fixes the transition and decay schedule before the data switch begins. The saved deadline is absolute: a restart does not create another 24 hours. Late recovery can require a limited overrun to retain a useful cooldown. Inspect the saved schedule and pipeline report when comparing requested and actual durations.

## 8. Data selection and preparation

Data preparation must make source selection, filtering, repetition, and evaluation overlap visible. Prefer a smaller inspected corpus to an opaque collection whose contents cannot be explained.

The preparation pipeline is:

1. Download the required source shards and supporting files into `raw/`.
2. Extract text or structured records with source-specific converters.
3. Filter malformed, excessively long, or unsuitable records and deduplicate where applicable.
4. Check overlap with held-out evaluation material using normalized 13-gram matching.
5. Allocate records according to the configured mixture, token budget, and repetition limits.
6. Encode documents with BOS boundaries and write `uint16` token shards, normally `2^24` tokens per shard.
7. Keep a fixed validation partition and record source counts, token totals, and rejection statistics.

Decontamination is an overlap filter, not proof that no leakage remains. Near duplicates and transformed benchmark questions require additional care. Evaluation examples must not be intentionally added to training.

| Source | Repository identifier | Intended use and handling |
| --- | --- | --- |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | General educational text; score-filtered pools for stages one and two |
| DCLM | `mlfoundations/dclm-baseline-1.0` | General English text |
| Cosmopedia v2 | `HuggingFaceTB/smollm-corpus` | Synthetic educational text; textbook subset for stage two |
| FineMath | `HuggingFaceTB/finemath` | `finemath-4plus` mathematical text |
| TinyGSM | `TinyGSM/TinyGSM` | Problem statements and executable Python solutions |
| Stack-Edu | `HuggingFaceTB/stack-edu` | Python metadata; source contents retrieved through Software Heritage |
| SmolTalk / SmolTalk2 | `HuggingFaceTB/smoltalk`, `HuggingFaceTB/smoltalk2` | Short dialogue and instruction examples |
| OpenMathInstruct-2 | `nvidia/OpenMathInstruct-2` | Short text solutions for stage two |
| OpenMathInstruct-1 | `nvidia/OpenMathInstruct-1` | Code-based solutions for tool SFT |
| GSM8K | `openai/gsm8k` | Training problems for conversion; held-out test problems for evaluation |
| MMLU | `cais/mmlu` | Auxiliary training data and held-out evaluation splits |
| ARC | `allenai/ai2_arc` | Training questions and held-out evaluation splits |
| HellaSwag | `Rowan/hellaswag` | Held-out completion evaluation |

Dataset identifiers describe the intended inputs. Record the actual revision, subset, downloaded files, and relevant usage terms in the experiment manifest; this table is not a license inventory. Python source can carry file-specific terms.

Inspect random decoded documents and confirm that boundaries, formats, and source proportions match expectations. Investigate a token-allocation deviation greater than 5%. A shortage of valid data should appear in the build report rather than being hidden by unlimited repetition.

## 9. Training stages and serialization

Pretraining and mid-training share one optimizer run: approximately 80% of its updates use stage one, followed by a transition and the final 20% on stage two. SFT is a separate stage with its own budget.

### Stage one: broad language pretraining

[`configs/stage1.yaml`](configs/stage1.yaml) requests 1.95B tokens and holds out 1% for validation. Shorter runs consume a shuffled prefix. Requested capacity does not guarantee that every pool has enough eligible records.

| Source | Token share | Maximum passes |
| --- | --- | --- |
| FineWeb-Edu | 40% | 1 |
| DCLM | 10% | 1 |
| Cosmopedia | 25% | 2 |
| FineMath | 10% | 2 |
| TinyGSM | 3% | 1 |
| Stack-Edu Python | 8% | 1 |
| SmolTalk dialogue | 4% | 1 |

For a mixture experiment, change one allocation at a time, keep the total at 100%, and rebuild the shards. Compare on a fixed validation set and the task relevant to the change.

### Stage two: instruction and tool exposure during decay

[`configs/stage2.yaml`](configs/stage2.yaml) requests 500M tokens, including capacity for the transition. General text remains in the mixture to reduce abrupt distribution changes.

| Source | Token share | Format |
| --- | --- | --- |
| Higher-quality FineWeb-Edu | 19.8% | Text |
| Pagoda identity text | 0.2% | Text |
| Cosmopedia textbooks | 15% | Text |
| TinyGSM | 25% | Dialogue with Python calls |
| OpenMathInstruct-2 | 9.6% | Dialogue with short written solutions |
| GSM8K training split | 0.4% | Dialogue with short written solutions |
| SmolTalk | 14.97% | Dialogue |
| Pagoda identity dialogue | 0.03% | Dialogue |
| MMLU auxiliary training and ARC training | 10% | Multiple choice |
| Stack-Edu Python | 5% | Code |

Identity examples should teach the model its name and scope without causing unrelated answers to become self-descriptions. Their recipes live under `identity/`.

### Stage three: supervised fine-tuning

[`configs/sft.yaml`](configs/sft.yaml) requests 109,700 conversations before filtering and availability limits:

| Category | Requested conversations |
| --- | --- |
| Everyday dialogue, short instructions, constraints, rewriting, and summarization | 60,200 |
| Python-assisted mathematics: GSM8K, OpenMathInstruct-1, and TinyGSM | 37,500 |
| Short explanatory answers | 10,000 |
| Pagoda identity | 2,000 |

These are requested counts, not the realized mixture. In particular, some tool sources contain fewer eligible unique examples. Report the built counts and token totals. Re-execute every tool example during the final SFT build and reject it if the result does not match its reference answer.

SFT uses assistant-token loss, excluding user text and executor-supplied tool outputs. Conversations are packed into 1,024-token training rows with masked padding. Defaults are 0.15 times the pretraining peak learning rates, 50 warmup steps, linear decay, 65,536 tokens per update, and at most two epochs. Select the saved epoch using validation loss; the deadline can also prevent a second epoch.

The serialized formats are:

```text
Plain document:
<|bos|>Document text...

Conversation:
<|bos|><|user_start|>User request<|user_end|><|assistant_start|>Assistant response<|assistant_end|>

Tool-assisted assistant response:
<|python_start|>
total = 3 * 12
print(total - 10)
<|python_end|><|output_start|>26<|output_end|>
Answer: 26<|assistant_end|>

Multiple-choice answer:
Question: ...
A. ... B. ... C. ... D. ...
Answer: B
```

The renderer inserts control tokens explicitly. It folds a leading system message into the first user message; there is no separate system-token pair in this vocabulary.

## 10. Execution workflow and monitoring

Preparation, throughput measurement, and a recovery smoke test precede the main run. For the longer budgets, the original planning allowance was about 55 or 100 GPU hours for pretraining and mid-training, leaving the remainder for preparation, SFT, evaluation, and interruptions. Recalculate this allowance from actual preparation costs.

Prepare the environment and data:

```bash
uv sync
uv run python -m scripts.download
uv run python -m scripts.prepare_data
uv run python -m scripts.tok_train --vocab 32768
uv run python -m scripts.build_shards
uv run python -m scripts.verify_data
```

Choose one training workflow. The launch scripts use the same `tmux` session name, so they are intended to run separately.

```bash
# 24-hour pipeline: pretraining, SFT, evaluation, and report
bash runs/train_24h.sh

# Three-day pretraining and mid-training
bash runs/train_3d.sh
RUN=d640_3d bash runs/after_pretrain.sh

# Five-day pretraining and mid-training
bash runs/train_5d.sh
RUN=d768_5d bash runs/after_pretrain.sh
```

For the longer workflows, run `after_pretrain.sh` after the training process has finished. Inspect each script and its entry point's `--help` before overriding defaults. The three- and five-day labels describe intended budgets; their default launch commands use token targets.

For the default 24-hour run:

```bash
tmux attach -t pagoda
tail -f logs/d512_24h/pipeline/pipeline.log
uv run python -m scripts.chat --run d512_24h --source sft --interactive
```

| Signal | Suggested review frequency | What to inspect |
| --- | --- | --- |
| Training loss, gradient norm, throughput, memory | Every 100 steps | Non-finite values, sustained drift, and throughput loss |
| Fixed validation loss and bits per byte | Every 500 steps | Trends over several measurements and source-distribution changes |
| Samples from fixed prompts | Every 2,000 steps | Coherence, repetition, format, and topic adherence |
| Base-model task evaluation | Around each quarter of progress | Trends with the same settings and sample selection |
| Checkpoint recovery | Before the main run and after relevant recovery changes | Restore weights, optimizer, schedule, and data position together |

The exact frequencies are configurable. Logs and checkpoint metadata should record what was actually used. Validation loss and benchmark scores need not improve at every observation. A missing immediate loss drop at the stage switch is a reason to inspect the transition, not proof that it failed.

Before accepting SFT, inspect at least 20 completions for malformed role boundaries, unclosed tool calls, repetition, and unsupported claims. Keep the prompts fixed across candidate checkpoints.

## 11. Python-assisted mathematics

The model's task is to translate a word problem into a short Python program. The executor performs the arithmetic and the model uses the returned result to finish its answer.

The generation protocol is:

1. Generate optional short explanatory text followed by `<|python_start|>`.
2. Generate code that prints the needed result, then `<|python_end|>`.
3. Stop generation and execute the code.
4. Insert the returned text between `<|output_start|>` and `<|output_end|>`.
5. Continue generation to `Answer: <number>` and `<|assistant_end|>`.

Allow at most two tool calls per response so the model can attempt one correction. Evaluation uses greedy generation, with a bounded output length. Numeric comparison accepts equivalent representations such as `26` and `26.0`; the implementation uses a tolerance scaled by `max(1, abs(reference))`.

The committed local executor starts isolated Python with `-I -S -B` in a temporary directory, closes standard input, applies a two-second wall-clock timeout, limits CPU time, and truncates returned output to 256 characters. A watchdog checks process-tree RSS against a 512 MB cap. This is a monitored limit rather than an instantaneous allocation barrier.

When available, the macOS `sandbox-exec` profile denies network access and writes outside the scratch directory. It permits host file reads, and the committed implementation can run without that profile when the facility is unavailable. Treat it as a local experimental executor, not a sufficient security boundary for arbitrary public input.

Data conversion must preserve the problem's meaning:

- Adapt TinyGSM's solution functions to the common printing protocol.
- Convert suitable GSM8K arithmetic annotations into executable assignments.
- Use OpenMathInstruct-1 code examples for tool SFT; use OpenMathInstruct-2 text solutions for the written-solution mixture.
- Execute converted examples and reject exceptions, non-numeric or ambiguous outputs, and mismatched answers.

Passing the execution filter establishes consistency with the reference answer, not that the program captures the intended reasoning for every possible input.

## 12. Evaluation and acceptance

Separate base-model evaluation from instruction-following and tool evaluation. Record the checkpoint, tokenizer, dataset split, number of examples, prompt format, decoding settings, and any context truncation alongside every score.

| Evaluation | Method | Primary purpose |
| --- | --- | --- |
| Validation loss and bits per byte | Fixed held-out text distribution | Compare training runs under the same data and tokenizer setup |
| CORE-style aggregate | Repository base-model evaluation bundle | Track broad completion performance; retain its task-level results |
| HellaSwag | Normalized completion likelihood, with up to 10 few-shot examples fitting the context | Evaluate continuation selection on base weights |
| ARC-Easy and ARC-Challenge | Multiple-choice evaluation | Inspect short question answering |
| MMLU | Multiple-choice evaluation | Characterize broad knowledge limitations |
| GSM8K with Python | Greedy tool-enabled generation and numeric answer checking | Measure problem-to-program success |
| HumanEval | Executed code evaluation | Inspect coding performance with appropriate isolation |
| Manual prompts | Fixed rubric and saved completions | Inspect behavior not summarized by a single score |

HellaSwag's few-shot setting is not ten-fold cross-validation. Do not compare scores obtained with different context fitting, task subsets, or checkpoint types as though they were identical protocols. The longer-run `after_pretrain.sh` script includes HumanEval; custom evaluation commands can select a different task list. Always report the tasks that actually ran.

The manual suite contains five prompts in each category:

- Writing: a short story, a science explanation, a formal rewrite, a continuation, and an outline.
- Instructions: exactly three points, a fixed output format, a length limit, an excluded word, and a two-step request.
- Questions: familiar knowledge, a common fact, an obscure fact, a definition, and a comparison.
- Mathematics: one-, two-, and three-step problems, a unit conversion, and a tool-error recovery case.

Use fixed sampling settings, such as temperature 0.7 and top-p 0.9, for manual comparisons. Evaluate each response against an explicit rubric and save failures. The obscure-fact case should test whether the model acknowledges uncertainty rather than merely rewarding a fluent answer.

Report actual results after a run. The original score ranges were hypotheses, not measured outcomes or hard limits imposed by the compute budget. Small score changes need scrutiny of sample count and uncertainty before being treated as progress.

## 13. Risks and recovery

| Risk | Evidence to look for | Response |
| --- | --- | --- |
| Sleep, power loss, or interruption | Missing process or a gap in logs | Restore the latest valid full checkpoint; verify the saved deadline |
| Thermal throttling or competing load | Sustained throughput decline relative to the benchmark | Check cooling and background activity; revise the remaining budget |
| Disk exhaustion | Failed downloads or checkpoint writes | Inspect storage, caches, and retention before removing expendable artifacts |
| Numerical instability | Non-finite values or persistent abnormal loss/gradient behavior | Preserve evidence, inspect data and precision, then use the recovery path |
| Stalled training | Flat validation trend across several evaluations | Check data position, repetition, learning-rate scaling, and distribution |
| Data-loader bottleneck | Low throughput despite stable compute conditions | Compare compute and loading costs; inspect shard access and preprocessing |
| Benchmark contamination | Unexpectedly strong scores or detected overlap | Audit inputs and rebuild clean data; invalid training may require retraining |
| Distribution shift at stage two | Sustained degradation after the switch | Verify the transition and mixture; compare task and general-text metrics |
| SFT overfitting | Validation loss rises while training loss falls | Select the better saved epoch and inspect repeated examples |
| Invalid tool syntax | Unclosed tokens, missing output, or failed execution | Check token IDs, rendering, loss masks, and converted examples |
| Wrong throughput projection | Completion estimate drifts beyond the budget | Replan before the decay schedule is frozen or revise the stated target |
| Excessive identity behavior | Unprompted self-description in ordinary answers | Inspect identity sampling and evaluate a reduced allocation |

Recovery actions should preserve the last known-good checkpoint and record what changed. A clean evaluation run cannot undo contamination already learned during training.

## 14. Optional experiments

Complete and evaluate the baseline before spending its budget on extensions. Each experiment needs a matched baseline, a success measure, and a stopping rule.

1. **Data-mixture comparisons.** Use `d256` and a fixed token budget to compare a few narrowly defined changes, such as more mathematical text or more dialogue. Keep validation data unchanged, then confirm a promising result at the target scale.
2. **Logit distillation.** A small teacher such as SmolLM2-135M is a possible starting point. Direct token-level KL requires compatible tokenization; adopting the teacher's vocabulary changes the experiment. Compare matched compute budgets, including teacher inference, before claiming a benefit.
3. **Tool-reward reinforcement learning.** An optional GSM8K experiment could reward verified final answers across multiple sampled solutions. This requires additional implementation and evaluation; benchmark-specific improvement does not establish general reasoning progress. Stop if the pilot shows no useful reward trend.
4. **Architecture variations.** Value embeddings, altered residual connections, or other changes should be isolated experiments after the baseline. Measure validation quality and wall-clock cost together.

No gain from these extensions is assumed in the baseline budget or expected results.

## 15. Accounting, reproducibility, and execution checklist

For rough compute accounting, the model implementation includes both block matrices and the vocabulary output projection:

```text
training_FLOPs ≈ D × [6 × (N_nonembed + V × d) + 12 × L × T × d]
```

Here `D` is the number of processed training tokens, `V` the vocabulary size, `L` the layer count, `T` the sequence length, and `d` the model width. This is an analytical estimate, not a measurement of device utilization or exact kernel work.

Approximate full-checkpoint state consists of fp32 parameters, Muon momentum for block matrices, and two AdamW states for embedding parameters, plus metadata. A tied 110M model is on the order of 1 GB per full checkpoint; measure the actual saved size and account for retained copies.

For each completed experiment, retain the source revision, dependency lockfile, model and optimizer settings, tokenizer report, data manifests, mixture files, seeds, recovery events, checkpoints, timing logs, evaluation outputs, and manual-test responses. Keep results separate from projections. Do not transplant fitted scaling-law loss values from another tokenizer or corpus into the results table.

- [ ] Power, cooling, free disk space, and sleep prevention are ready.
- [ ] Dependencies are installed and a small model completes a training smoke test.
- [ ] Saving, loading, and resuming a full checkpoint have been exercised.
- [ ] The local executor handles normal output, timeout, exception, missing output, and multiple outputs as expected.
- [ ] Sustained throughput and memory use have been measured on real data.
- [ ] Model size, token budget, and recovery reserve match the measured time budget.
- [ ] Dataset revisions, input files, filtering decisions, and rejection rates are recorded.
- [ ] Held-out evaluation overlap checks and train/validation separation have been reviewed.
- [ ] Tokenizer acceptance checks pass and the tokenizer is frozen.
- [ ] Shard counts, source proportions, document boundaries, and decoded samples have been inspected.
- [ ] Fixed sampling prompts and the manual rubric are saved before training.
- [ ] The main run produces usable logs and a recoverable checkpoint.
- [ ] The stage-two transition and learning-rate decay occur at the saved schedule positions.
- [ ] Tool SFT examples execute successfully and match their reference answers.
- [ ] SFT selection uses validation results and respects the available budget.
- [ ] Base-model, instruction, and tool evaluations are labeled separately.
- [ ] Final reports contain measured results, failure examples, and unresolved limitations.
