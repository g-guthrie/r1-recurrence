#!/usr/bin/env python3
import os
os.environ.setdefault("HF_HOME", "/workspace/hf-cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/hf-cache/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/hf-cache/transformers")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_ENABLE_XET"] = "0"
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from recurrence_r1_core import RecurrenceConfig, generate_with_recurrence


def pick_device_dtype():
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return "cuda", (torch.bfloat16 if major >= 8 else torch.float16)
    return "cpu", torch.float32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument(
        "--prompt",
        default="Explain recursion in one sentence. End with: Final answer: <statement>.",
    )
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0, help="Set >0 for deterministic torch seeds")
    # Recurrence knobs
    ap.add_argument("--steps", type=int, default=0, help="0 = baseline (no recurrence)")
    ap.add_argument("--entropy_gate", type=float, default=1.3)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--margin_gate", type=float, default=0.25)
    ap.add_argument("--time_limit_s", type=float, default=12.0)
    ap.add_argument("--max_branches", type=int, default=4)
    ap.add_argument("--only_answer_phase", action="store_true", default=True)
    ap.add_argument("--no_only_answer_phase", dest="only_answer_phase", action="store_false")
    ap.add_argument("--ban_meta_placeholders", action="store_true", default=True)
    ap.add_argument("--no_ban_meta_placeholders", dest="ban_meta_placeholders", action="store_false")
    # RRH-specific knobs
    ap.add_argument("--rrh_alpha", type=float, default=0.75)
    ap.add_argument("--rrh_tau", type=float, default=0.5)
    ap.add_argument("--rrh_clip", type=float, default=2.0)
    ap.add_argument("--rrh_token_time_ms", type=float, default=35.0)
    ap.add_argument("--rrh_cooloff_tokens", type=int, default=24)
    ap.add_argument("--rrh_min_improve_eps", type=float, default=1e-4)
    args = ap.parse_args()

    # Seeding for reproducibility
    if args.seed and args.seed > 0:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device, dtype = pick_device_dtype()
    print(f"Loading {args.model} on {device} ({dtype}) ...")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto",
    )
    model.eval()

    if args.steps <= 0:
        # Baseline
        enc = tok(args.prompt, return_tensors="pt").to(model.device)
        do_sample = args.temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
            "pad_token_id": tok.eos_token_id,
        }
        if do_sample:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": args.temperature,
                }
            )
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                **gen_kwargs,
            )
        text = tok.decode(out_ids[0], skip_special_tokens=True)
    else:
        # Recurrence first-pass
        cfg = RecurrenceConfig(
            steps=args.steps,
            entropy_gate=args.entropy_gate,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            time_limit_s=args.time_limit_s,
            max_branches=args.max_branches,
            margin_gate=args.margin_gate,
            only_answer_phase=args.only_answer_phase,
            ban_meta_placeholders=args.ban_meta_placeholders,
            rrh_alpha=args.rrh_alpha,
            rrh_tau=args.rrh_tau,
            rrh_clip=args.rrh_clip,
            rrh_token_time_ms=args.rrh_token_time_ms,
            rrh_cooloff_tokens=args.rrh_cooloff_tokens,
            rrh_min_improve_eps=args.rrh_min_improve_eps,
        )
        text = generate_with_recurrence(model, tok, args.prompt, cfg)

    print("\n--- OUTPUT ---\n")
    print(text)
    print("\n-------------\n")


if __name__ == "__main__":
    main()
