"""Reward-path diagnostic for the train-reward-≡0 anomaly (experiments.md D9 followup).

Run `20260601T032731Z` scored mean_reward=0.0 on every training step while
eval scored pass@1=0.625 — same model, same generate engine. The two variables
that differ are temperature (train 1.0 vs eval greedy) and the gold-answer path
(train = hendrycks_math boxed-extraction; eval = MATH-500 curated). This script
isolates them: for a few hendrycks_math L1-3 prompts (the training
distribution), it generates GREEDY and TEMPERATURE completions, scores each with
the same math-verify logic OutcomeRewardModule uses, and prints golds +
completion tails.

Read the summary line:
  - greedy correct, temp wrong  -> H1 (temperature): base model degenerates at
    temp 1.0; lower the training temperature (and/or few-shot the base model).
  - greedy ALSO wrong           -> H2 (gold/format): the boxed-extracted gold
    answers don't verify even against good completions; fix the reward path.

Run:  modal run modal_app/reward_debug.py            # defaults: 5 prompts, temp 1.0
      modal run modal_app/reward_debug.py --temperature 0.7 --num-prompts 8
"""

from __future__ import annotations

import modal

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "torch>=2.4",
        "transformers>=4.50",
        "accelerate>=1.0",
        "datasets>=2.20",
        "math-verify>=0.5",
        "sentencepiece",
    )
    .add_local_python_source("lora_reward_density")
)

hf_cache = modal.Volume.from_name("lrd-hf-cache", create_if_missing=True)
app = modal.App("lora-reward-density-reward-debug", image=image)


@app.function(gpu="A10G", timeout=1800, volumes={"/hf-cache": hf_cache})
def debug(
    *,
    model_id: str,
    num_prompts: int,
    levels: tuple[int, ...],
    temperature: float,
    n_sampled: int,
    max_tokens: int,
    train_mode: bool,
) -> None:
    import contextlib
    import io
    import os

    os.environ["HF_HOME"] = "/hf-cache/huggingface"
    os.environ["HF_HUB_CACHE"] = "/hf-cache/huggingface/hub"

    import torch
    from math_verify import parse, verify
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lora_reward_density.data import load_math_train

    examples = load_math_train(
        num_examples=num_prompts,
        levels=list(levels),
        seed=0,
        cache_dir="/hf-cache/datasets",
    )

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    # train_mode reproduces the training rollout, which calls generate while the
    # student is in train() mode (eval + the diagnostic use eval mode). If this
    # tanks correctness vs eval mode, the bug is rolling out in train mode
    # (dropout noise corrupting sampling) — fix: generate under eval mode.
    if train_mode:
        model.train()
        print(
            f"[mode] train() — config attention_dropout="
            f"{getattr(model.config, 'attention_dropout', 'n/a')}"
        )
    else:
        model.eval()
        print("[mode] eval()")

    def score(gold: str, completion: str) -> tuple[bool, bool, bool]:
        """Mirror OutcomeRewardModule: parse gold + completion, verify. Returns
        (correct, gold_parsed_nonempty, ans_parsed_nonempty)."""
        with contextlib.redirect_stderr(io.StringIO()):
            g = parse(gold, parsing_timeout=5)
            a = parse(completion, parsing_timeout=5)
        ok = bool(a) and bool(verify(g, a, timeout_seconds=5))
        return ok, bool(g), bool(a)

    def generate(prompt: str, do_sample: bool, n: int) -> list[str]:
        enc = tok([prompt] * n, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=(temperature if do_sample else None),
                top_p=(0.95 if do_sample else None),
                pad_token_id=tok.pad_token_id,
            )
        return tok.batch_decode(out[:, enc.input_ids.shape[1] :], skip_special_tokens=True)

    greedy_correct = 0
    sampled_correct = 0
    sampled_total = 0
    for i, ex in enumerate(examples):
        gold = ex.metadata["gold_answer"]
        print(f"\n===== prompt {i} (level {ex.metadata.get('level')}) =====")
        print("PROMPT:", ex.prompt[:200].replace("\n", " "))
        print("GOLD:", repr(gold))

        g_comp = generate(ex.prompt, do_sample=False, n=1)[0]
        ok, gp, ap = score(gold, g_comp)
        greedy_correct += int(ok)
        has_box = "\\boxed" in g_comp
        print(f"[greedy] correct={ok} gold_parsed={gp} ans_parsed={ap} has_boxed={has_box}")
        print("  tail:", repr(g_comp[-250:]))

        for j, s_comp in enumerate(generate(ex.prompt, do_sample=True, n=n_sampled)):
            ok, _gp, ap = score(gold, s_comp)
            sampled_correct += int(ok)
            sampled_total += 1
            has_box = "\\boxed" in s_comp
            print(f"[temp={temperature} #{j}] correct={ok} ans_parsed={ap} has_boxed={has_box}")
            print("  tail:", repr(s_comp[-200:]))

    print("\n===== SUMMARY =====")
    print(f"greedy:        {greedy_correct}/{len(examples)} correct")
    print(f"temp={temperature}:    {sampled_correct}/{sampled_total} correct")
    print(
        "Interpretation: greedy-correct + temp-wrong -> H1 (lower training temp); "
        "greedy also wrong -> H2 (gold/format verify mismatch)."
    )


@app.local_entrypoint()
def main(
    model_id: str = "Qwen/Qwen3-1.7B-Base",
    num_prompts: int = 5,
    levels: str = "1,2,3",
    temperature: float = 1.0,
    n_sampled: int = 4,
    max_tokens: int = 1024,
    train_mode: bool = False,
) -> None:
    parsed = tuple(int(x.strip()) for x in levels.split(","))
    debug.remote(
        model_id=model_id,
        num_prompts=num_prompts,
        levels=parsed,
        temperature=temperature,
        n_sampled=n_sampled,
        max_tokens=max_tokens,
        train_mode=train_mode,
    )
