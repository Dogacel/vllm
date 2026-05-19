# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# spec_decode_gptoss_longctx.py
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

TARGET = "openai/gpt-oss-120b"
DRAFT = "Dogacel/specdrift-gpt-oss-120b-eagle3"
# DRAFT = "nvidia/gpt-oss-120b-Eagle3-v3"

FILLER_DATASET = "HuggingFaceH4/ultrachat_200k"
FILLER_SPLIT = "train_sft"

from vllm.config.speculative import SpeculativeConfig

_orig = SpeculativeConfig.hf_config_override


@staticmethod
def _with_swa(hf_config):
    hf_config = _orig(hf_config)
    hf_config.sliding_window = 1
    hf_config.use_sliding_window = True
    hf_config.layer_types = ["sliding_attention"]  # 1 entry per draft layer
    return hf_config


SpeculativeConfig.hf_config_override = _with_swa


def build_filler_turns(tok, target_tokens: int, seed: int = 0):
    """Stream multi-turn chats until we approximately hit target_tokens.

    Ends on an assistant turn so the real MT-Bench user question follows cleanly.
    Uses a cheap per-turn token estimate (no repeated chat-template renders).
    """
    if target_tokens <= 0:
        return []

    stream = load_dataset(FILLER_DATASET, split=FILLER_SPLIT, streaming=True).shuffle(
        seed=seed, buffer_size=1000
    )

    msgs, approx = [], 0
    for ex in stream:
        for turn in ex["messages"]:
            if turn["role"] not in ("user", "assistant"):
                continue
            # +8 is a rough per-turn chat-template overhead; fine for sizing.
            approx += len(tok(turn["content"], add_special_tokens=False).input_ids) + 8
            msgs.append({"role": turn["role"], "content": turn["content"]})
            if approx >= target_tokens:
                while msgs and msgs[-1]["role"] != "assistant":
                    msgs.pop()
                return msgs
    return msgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_spec_tokens", type=int, default=5)
    p.add_argument("--num_prompts", type=int, default=20)
    p.add_argument("--max_num_seqs", type=int, default=8)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument(
        "--ctx_tokens",
        type=int,
        default=32000,
        help="Approx. filler-history tokens to prepend to each MT-Bench prompt.",
    )
    p.add_argument("--max_model_len", type=int, default=40960)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(TARGET)
    ds = load_dataset("philschmid/mt-bench", split="train")

    prompts = []
    for i, ex in enumerate(ds.select(range(args.num_prompts))):
        history = build_filler_turns(tok, args.ctx_tokens, seed=i)
        messages = history + [{"role": "user", "content": ex["turns"][0]}]
        prompts.append(
            tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    lens = [len(tok(p).input_ids) for p in prompts]
    print(
        f"prompt token lens: min={min(lens)} max={max(lens)} "
        f"mean={sum(lens) / len(lens):.0f}  (target={args.ctx_tokens})"
    )

    llm = LLM(
        model=TARGET,
        tensor_parallel_size=args.tp,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_log_stats=False,
        speculative_config={
            "method": "eagle3",
            "model": DRAFT,
            "num_speculative_tokens": args.num_spec_tokens,
        },
    )

    sp = SamplingParams(temperature=0.7, max_tokens=args.max_tokens)
    llm.generate(prompts, sp)

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
    print(f"ctx_tokens (target) : {args.ctx_tokens}")
    print(f"batch (max_num_seqs): {args.max_num_seqs}")
    print(f"num_drafts          : {num_drafts}")
    print(f"mean acceptance len : {mean_al:.3f}")
    if per_pos:
        for i, c in enumerate(per_pos):
            rate = c / max(num_drafts, 1)
            print(f"  accept@pos {i}      : {rate:.3f}  ({c})")


if __name__ == "__main__":
    main()
