"""What we download from the Hugging Face Hub (plan sections 8 and 15).

Everything is fetched file-by-file with `hf_hub_download(local_dir=...)` so the
global ~/.cache/huggingface never holds an Arrow copy. File counts are sized so
that the largest budget the day-0 benchmark allows (2.4B tokens: 1.95B stage 1 +
0.5B stage 2, d768 for 100 GPU hours at ~7.8k tok/s) fits inside each source's
epoch cap; smaller runs consume a prefix of the same shuffled shards.
"""

# name -> (repo_id, [files]); everything lands in raw/<name>/<file path>
HF_FILES = {
    # ~0.73B GPT-2 tokens per file. 000 feeds stage 1 (score>=3); 001's score>=4
    # docs are the stage-2 pool and its score-3 docs top up stage 1.
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", [
        "sample/10BT/000_00000.parquet",
        "sample/10BT/001_00000.parquet",
    ]),
    # ~135M tokens per jsonl.zst file
    "dclm": ("mlfoundations/dclm-baseline-1.0", [
        "global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst",
        "global-shard_01_of_10/local-shard_0_of_10/shard_00000001_processed.jsonl.zst",
        "global-shard_01_of_10/local-shard_0_of_10/shard_00000002_processed.jsonl.zst",
        "global-shard_01_of_10/local-shard_0_of_10/shard_00000003_processed.jsonl.zst",
    ]),
    # ~270M tokens per file; 'format' column carries textbook / story / blogpost ...
    "cosmopedia_v2": ("HuggingFaceTB/smollm-corpus", [
        "cosmopedia-v2/train-00000-of-00104.parquet",
        "cosmopedia-v2/train-00001-of-00104.parquet",
        "cosmopedia-v2/train-00002-of-00104.parquet",
    ]),
    # ~150M tokens per file
    "finemath": ("HuggingFaceTB/finemath", [
        "finemath-4plus/train-00000-of-00064.parquet",
        "finemath-4plus/train-00001-of-00064.parquet",
    ]),
    # ~700k problems per file (question + def simple_math_problem() code)
    "tinygsm": ("TinyGSM/TinyGSM", [
        "data/train-00000-of-00017.parquet",
        "data/train-00001-of-00017.parquet",
        "data/train-00002-of-00017.parquet",
    ]),
    # Stack-Edu only ships blob ids; contents are fetched from Software Heritage
    # (scripts.download --stack-edu) into raw/stack_edu/python_files.jsonl
    "stack_edu": ("HuggingFaceTB/stack-edu", [
        "Python/train-00000-of-00005.parquet",
    ]),
    # SmolTalk v1: 'all' feeds the chat-format text of stage 1/2; smol-constraints feeds SFT
    "smoltalk": ("HuggingFaceTB/smoltalk", [
        "data/all/train-00000-of-00009.parquet",
        "data/all/train-00001-of-00009.parquet",
        "data/all/train-00002-of-00009.parquet",
        "data/all/train-00003-of-00009.parquet",
        "data/all/train-00004-of-00009.parquet",
        "data/all/train-00005-of-00009.parquet",
        "data/all/train-00006-of-00009.parquet",
        "data/smol-constraints/train-00000-of-00001.parquet",
    ]),
    # SmolTalk2 non-reasoning subsets for SFT
    "smoltalk2": ("HuggingFaceTB/smoltalk2", [
        "SFT/smoltalk_smollm3_everyday_conversations_no_think-00000-of-00001.parquet",
        "SFT/smoltalk_smollm3_smol_magpie_ultra_no_think-00000-of-00006.parquet",
        "SFT/smoltalk_smollm3_smol_magpie_ultra_no_think-00001-of-00006.parquet",
        "SFT/smoltalk_smollm3_smol_magpie_ultra_no_think-00002-of-00006.parquet",
        "SFT/smoltalk_smollm3_smol_rewrite_no_think-00000-of-00001.parquet",
        "SFT/smoltalk_smollm3_smol_summarize_no_think-00000-of-00001.parquet",
    ]),
    # text-only CoT solutions (no code), used for stage-2 short CoT
    # about half of the solutions exceed the 300-token cap, hence two files
    "openmathinstruct2": ("nvidia/OpenMathInstruct-2", [
        "data/train_1M-00000-of-00003.parquet",
        "data/train_1M-00001-of-00003.parquet",
        "data/train_1M-00002-of-00003.parquet",
    ]),
    "gsm8k": ("openai/gsm8k", [
        "main/train-00000-of-00001.parquet",
        "main/test-00000-of-00001.parquet",
    ]),
    "mmlu": ("cais/mmlu", [
        "auxiliary_train/train-00000-of-00001.parquet",
        "all/test-00000-of-00001.parquet",
        "all/dev-00000-of-00001.parquet",
        "all/validation-00000-of-00001.parquet",
    ]),
    "arc": ("allenai/ai2_arc", [
        "ARC-Easy/train-00000-of-00001.parquet",
        "ARC-Easy/validation-00000-of-00001.parquet",
        "ARC-Easy/test-00000-of-00001.parquet",
        "ARC-Challenge/train-00000-of-00001.parquet",
        "ARC-Challenge/validation-00000-of-00001.parquet",
        "ARC-Challenge/test-00000-of-00001.parquet",
    ]),
    "hellaswag": ("Rowan/hellaswag", [
        "data/train-00000-of-00001.parquet",       # few-shot examples only
        "data/validation-00000-of-00001.parquet",
    ]),
    "humaneval": ("openai/openai_humaneval", [
        "openai_humaneval/test-00000-of-00001.parquet",
    ]),
}

# OpenMathInstruct-2 has no code solutions, so the tool-format math in SFT
# (plan section 11) uses OpenMathInstruct-1's code-interpreter solutions instead.
# The 1.3GB jsonl is read as a stream and only GSM8K-derived rows are kept.
OMI1_REPO = "nvidia/OpenMathInstruct-1"
OMI1_FILE = "correct_solutions/train.jsonl"
OMI1_MAX_ROWS = 60_000

# Stack-Edu Python contents (Software Heritage public S3 bucket, gzip blobs)
SWH_URL = "https://softwareheritage.s3.amazonaws.com/content/{blob_id}"
STACK_EDU_TARGET_BYTES = 950_000_000   # ~185M tokens needed (stage 1 + 2 at the 2.4B budget) at ~3.5 B/token
STACK_EDU_MAX_BYTES = 32_000            # files > ~8k tokens are dropped anyway

# CORE eval bundle used by nanochat (DCLM CORE tasks)
CORE_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
