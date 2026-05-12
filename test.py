# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# spec_decode_gptoss.py
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

TARGET = "openai/gpt-oss-120b"
# DRAFT  = "RedHatAI/gpt-oss-120b-speculator.eagle3"
DRAFT = "nvidia/gpt-oss-120b-Eagle3-v3"
# DRAFT = "Dogacel/specdrift-gpt-oss-120b-eagle3"

# TARGET = "openai/gpt-oss-20b"
# DRAFT  = "RedHatAI/gpt-oss-20b-speculator.eagle3"
# DRAFT = "Dogacel/specdrift-gpt-oss-20b-eagle3"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_spec_tokens", type=int, default=5)
    p.add_argument("--num_prompts", type=int, default=20)
    p.add_argument("--max_num_seqs", type=int, default=8)
    p.add_argument(
        "--tp", type=int, default=1
    )  # 120B fits on 1xH100 (MXFP4); bump to 2/4 for KV headroom
    p.add_argument("--max_tokens", type=int, default=2048)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(TARGET)
    ds = load_dataset("philschmid/mt-bench", split="train")
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": ex["turns"][0]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for ex in ds.select(range(args.num_prompts))
    ]

    llm = LLM(
        model=TARGET,
        tensor_parallel_size=1,
        max_num_seqs=8,  # ← drop way down
        max_model_len=4096,  # ← drop further
        gpu_memory_utilization=0.95,  # ← push higher
        enforce_eager=True,  # ← per the RedHatAI 120b card
        enable_prefix_caching=False,
        disable_log_stats=False,
        speculative_config={
            "method": "eagle3",
            "model": DRAFT,
            "num_speculative_tokens": 5,
        },
    )

    sp = SamplingParams(temperature=0.7, max_tokens=args.max_tokens)
    llm.generate(prompts, sp)

    # Pull V1 spec-decode metrics
    metrics = llm.get_metrics()
    num_drafts = num_accepted = 0
    per_pos = None
    for m in metrics:
        if isinstance(m, Counter) and m.name == "vllm:spec_decode_num_drafts":
            num_drafts = m.value
        elif (
            isinstance(m, Counter) and m.name == "vllm:spec_decode_num_accepted_tokens"
        ):
            num_accepted = m.value
        elif (
            isinstance(m, Vector)
            and m.name == "vllm:spec_decode_num_accepted_tokens_per_pos"
        ):
            per_pos = m.values

    mean_al = 1 + num_accepted / max(num_drafts, 1)
    print("-" * 60)
    print(f"num_spec_tokens     : {args.num_spec_tokens}")
    print(f"batch (max_num_seqs): {args.max_num_seqs}")
    print(f"num_drafts          : {num_drafts}")
    print(f"mean acceptance len : {mean_al:.3f}")
    if per_pos:
        for i, c in enumerate(per_pos):
            rate = c / max(num_drafts, 1)
            print(f"  accept@pos {i}      : {rate:.3f}  ({c})")


if __name__ == "__main__":
    main()
