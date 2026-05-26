"""Smoke test on Modal: load Qwen3-1.7B-Base on an H100, generate one completion, dump logprobs.

Run:
    modal run modal_app/smoke_test.py

What this validates:
    - GPU access on Modal (H100)
    - HF model download + bf16 load
    - Greedy generate + per-token logprob extraction

What this does NOT exercise (intentionally — Tier 1 work):
    - vLLM, LoRA, GRPO, reward modules.
"""

from __future__ import annotations

import modal

MODEL_ID = "Qwen/Qwen3-1.7B-Base"
PROMPT = (
    "Problem: A train leaves Station A at 9:00 AM traveling at 60 mph. "
    "Another train leaves Station B at 10:00 AM traveling at 80 mph toward Station A. "
    "If A and B are 300 miles apart, when do the trains meet?\nSolution:"
)
MAX_NEW_TOKENS = 64

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.4",
    "transformers>=4.50",
    "accelerate>=1.0",
    "sentencepiece",
)

app = modal.App("lora-reward-density-smoke", image=image)


@app.function(gpu="H100", timeout=900)
def smoke() -> dict:
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    info: dict = {
        "cuda_available": torch.cuda.is_available(),
        "device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "torch_version": torch.__version__,
    }

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    info["load_seconds"] = round(time.perf_counter() - t0, 2)

    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    info["generate_seconds"] = round(time.perf_counter() - t0, 2)

    completion_ids = out.sequences[0, inputs.input_ids.shape[1] :]
    info["completion"] = tokenizer.decode(completion_ids, skip_special_tokens=True)
    info["completion_length"] = int(completion_ids.shape[0])

    logprobs: list[float] = []
    for step_logits, tok in zip(out.scores, completion_ids.tolist(), strict=True):
        log_probs = torch.log_softmax(step_logits[0].float(), dim=-1)
        logprobs.append(log_probs[tok].item())
    info["sum_logprob"] = round(sum(logprobs), 4)
    info["mean_logprob"] = round(sum(logprobs) / len(logprobs), 4)
    info["first_5_logprobs"] = [round(x, 4) for x in logprobs[:5]]

    return info


@app.local_entrypoint()
def main() -> None:
    import json

    result = smoke.remote()
    print(json.dumps(result, indent=2))
