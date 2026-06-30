# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.model_loader import get_model


def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from vllm.compilation.backends import set_model_tag

    # DSpark uses non-causal attention.
    causal = False
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=not causal,
            backend=speculative_config.attention_backend,
        ),
    )

    with set_model_tag("dspark_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    if get_pp_group().world_size != 1:
        raise NotImplementedError("DSpark does not support pipeline parallelism.")

    # Alias the target's embed_tokens / lm_head for any the draft does not ship.
    # DeepSeek-V4 drafts share both; dense Qwen3 drafts share only what is absent.
    shares_all = getattr(draft_model, "dspark_shares_target_embeddings", True)
    includes_lm_head = getattr(draft_model, "_draft_includes_lm_head", True)
    includes_embed_tokens = getattr(draft_model, "_draft_includes_embed_tokens", True)
    share_embed = shares_all or not includes_embed_tokens
    share_lm_head = shares_all or not includes_lm_head
    if not share_embed and not share_lm_head:
        return draft_model

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model

    # Share the vocab embedding (target.model.embed_tokens -> draft.model).
    if share_embed:
        target_embed = getattr(target_inner, "embed_tokens", None)
        if target_embed is not None:
            if getattr(draft_inner, "embed_tokens", None) is not None:
                del draft_inner.embed_tokens
            draft_inner.embed_tokens = target_embed

    # Share the LM head (target.lm_head -> draft.lm_head).
    if share_lm_head:
        target_lm_head = getattr(target_model, "lm_head", None)
        if target_lm_head is not None:
            if getattr(draft_model, "lm_head", None) is not None:
                del draft_model.lm_head
            draft_model.lm_head = target_lm_head

    return draft_model
