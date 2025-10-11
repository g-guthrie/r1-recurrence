import os
os.environ.setdefault("HF_HOME", "/workspace/hf-cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/hf-cache/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/hf-cache/transformers")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_ENABLE_XET"] = "0"
import argparse, json, time, re, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from recurrence_r1_core import RecurrenceConfig, refine_tail_then_generate

YES_WORDS = {"yes", "y", "true"}
NO_WORDS = {"no", "n", "false"}
PARITY_WORDS = {"even", "odd"}
BOX_CODES = {"gg", "gs", "ss"}
WEEKDAY_MAP = {
    "monday": "monday",
    "tuesday": "tuesday",
    "wednesday": "wednesday",
    "thursday": "thursday",
    "friday": "friday",
    "saturday": "saturday",
    "sunday": "sunday",
}

# Force one-line final answer per task for evaluation stability
FORMAT = {
    "divisibility": "<yes/no>",
    "parity": "<even/odd>",
    "algebra": "<number>",
    "sum-digits": "<number>",
    "prob": "<a/b>",
    "date": "<weekday>",
    "logic": "<yes/no>",
    "coins-boxes": "<box>",
}

def wrap_for_eval(ex_prompt: str, task_id: str) -> str:
    tag = FORMAT.get(task_id, "<answer>")
    return (
        ex_prompt.rstrip()
        + "\n\nReply with a single line only:\n"
        + f"Final answer: {tag}\n"
        + "Do not include any explanation or intermediate steps."
    )

def load_model(model_id, dtype="bfloat16", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if dtype == "bfloat16" and torch.cuda.is_available() else torch.float16
    print(f"Loading {model_id} on {device} ({dt}) ...")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dt,
        device_map="auto"
    )
    model.eval()
    return tok, model, device, dt

def extract_final(text: str):
    # Preferred: parse explicit final-answer line
    m = re.search(r"final answer:\s*([^\r\n]+)", text, flags=re.IGNORECASE)
    if m:
        ans = m.group(1).strip()
        ans = re.sub(r"^[<\s]*", "", ans)
        ans = re.sub(r"[>\s.。\u3002]+$", "", ans)
        return ans.lower()
    # Fallbacks over entire text
    t = text.lower()
    for pat in [r"\b(yes|no)\b", r"\b(even|odd)\b", r"\b(gg|gs|ss)\b"]:
        m = re.search(pat, t)
        if m:
            return m.group(1)
    m = re.search(r"-?\d+\s*/\s*-?\d+", t)
    if m:
        return re.sub(r"\s+", "", m.group(0))
    m = re.search(r"-?\d+", t)
    if m:
        try:
            return str(int(m.group(0)))
        except Exception:
            return m.group(0)
    toks = re.findall(r"[A-Za-z0-9/]+", t)
    return toks[-1] if toks else ""


def canonicalize_answer(ans: str) -> str:
    original = ans
    ans = ans.strip().lower()
    ans = re.sub(r"\\boxed\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"<[^>]+>", "", ans)
    ans = re.sub(r"[,\s]+", " ", ans).strip()

    if ans in YES_WORDS:
        return "yes"
    if ans in NO_WORDS:
        return "no"
    if ans in PARITY_WORDS:
        return ans
    if ans in WEEKDAY_MAP:
        return WEEKDAY_MAP[ans]

    compact = ans.replace(" ", "")
    for code in BOX_CODES:
        if code in compact:
            return code

    frac_match = re.search(r"-?\d+/\d+", ans)
    if frac_match:
        return frac_match.group(0)

    int_match = re.search(r"-?\d+", ans)
    if int_match:
        return str(int(int_match.group(0)))

    if ans:
        return ans

    fallback = re.findall(r"[a-z0-9/]+", original.lower())
    return fallback[-1] if fallback else ""


def normalize_target(t: str):
    cleaned = re.sub(r"\\boxed\{([^}]*)\}", r"\1", t)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = cleaned.strip().lower()
    cleaned = re.sub(r"[,\s]+", " ", cleaned).strip()
    return canonicalize_answer(cleaned)

def run_one(tok, model, device, prompt, max_new_tokens, temperature, cfg: RecurrenceConfig):
    enc = tok(prompt, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    if cfg.steps > 0:
        full = refine_tail_then_generate(
            model,
            tok,
            prompt,
            cfg,
            device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        out = full[len(prompt):].lstrip() if full.startswith(prompt) else full
    else:
        do_sample = temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tok.eos_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        gen = model.generate(
            **enc,
            **gen_kwargs,
        )
        continuation_ids = gen[0][enc["input_ids"].shape[1]:]
        out = tok.decode(continuation_ids, skip_special_tokens=True)
    t1 = time.perf_counter()
    return out, (t1 - t0)

def evaluate(path_jsonl, tok, model, device, max_new_tokens, temperature, cfg: RecurrenceConfig):
    total, correct, rows = 0, 0, []
    with open(path_jsonl, "r") as f:
        for line in f:
            if not line.strip(): continue
            ex = json.loads(line)
            total += 1
            wrapped = wrap_for_eval(ex["prompt"], ex["id"])
            out, secs = run_one(tok, model, device, wrapped, max_new_tokens, temperature, cfg)
            pred = canonicalize_answer(extract_final(out))
            gold = normalize_target(ex["target"])
            ok = (pred == gold)
            correct += int(ok)
            rows.append({"id": ex["id"], "ok": ok, "pred": pred, "gold": gold, "secs": round(secs, 3)})
            print(f"[{ex['id']}] ok={ok} pred='{pred}' gold='{gold}' time={secs:.2f}s")
    acc = correct / max(1, total)
    return acc, rows

def save_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--eval_file", default="evals/micro_reasoning.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="Evaluate only first N examples if >0")
    ap.add_argument("--save_texts", default="", help="Optional path to save raw decoded texts JSONL")
    ap.add_argument("--dump_config", default="", help="Optional path to dump run config JSON")
    ap.add_argument("--steps", type=int, default=0)
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
    ap.add_argument("--out", default="outputs/run.jsonl")
    args = ap.parse_args()

    # Seeding
    if args.seed and args.seed > 0:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    tok, model, device, dt = load_model(args.model)
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
    print(f"Recurrence {'ON' if cfg.steps>0 else 'OFF'} | steps={cfg.steps} gate={cfg.entropy_gate}")
    # Optionally limit eval size by creating a temp filtered file
    eval_file = args.eval_file
    if args.limit and args.limit > 0:
        import json, tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8")
        with open(args.eval_file, "r", encoding="utf-8") as src:
            for i, line in enumerate(src):
                if not line.strip():
                    continue
                tmp.write(line)
                if i + 1 >= args.limit:
                    break
        tmp.close()
        eval_file = tmp.name

    acc, rows = evaluate(eval_file, tok, model, device, args.max_new_tokens, args.temperature, cfg)
    print(f"\nAccuracy: {acc*100:.1f}%  ({sum(r['ok'] for r in rows)}/{len(rows)})")
    save_jsonl(args.out, rows)
    print(f"Wrote {args.out}")

    # Optional artifacts
    if args.save_texts:
        import json as _json
        os.makedirs(os.path.dirname(args.save_texts) or ".", exist_ok=True)
        with open(eval_file, "r", encoding="utf-8") as src, open(args.save_texts, "w", encoding="utf-8") as dst:
            for line in src:
                if not line.strip():
                    continue
                ex = _json.loads(line)
                text, secs = run_one(tok, model, device, wrap_for_eval(ex["prompt"], ex["id"]), args.max_new_tokens, args.temperature, cfg)
                dst.write(_json.dumps({"id": ex["id"], "text": text, "secs": round(secs, 3)}) + "\n")
        print(f"Wrote {args.save_texts}")
    if args.dump_config:
        import json as _json
        payload = {
            "model": args.model,
            "device": device,
            "dtype": str(dt),
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "eval_file": args.eval_file,
            "limit": args.limit,
            "recurrence": {
                "steps": args.steps,
                "entropy_gate": args.entropy_gate,
                "top_k": args.top_k,
                "margin_gate": args.margin_gate,
                "time_limit_s": args.time_limit_s,
                "max_branches": args.max_branches,
                "only_answer_phase": args.only_answer_phase,
                "ban_meta_placeholders": args.ban_meta_placeholders,
                "rrh_alpha": args.rrh_alpha,
                "rrh_tau": args.rrh_tau,
                "rrh_clip": args.rrh_clip,
                "rrh_token_time_ms": args.rrh_token_time_ms,
                "rrh_cooloff_tokens": args.rrh_cooloff_tokens,
                "rrh_min_improve_eps": args.rrh_min_improve_eps,
            },
            "out": args.out,
        }
        os.makedirs(os.path.dirname(args.dump_config) or ".", exist_ok=True)
        with open(args.dump_config, "w", encoding="utf-8") as h:
            _json.dump(payload, h, indent=2)
        print(f"Wrote {args.dump_config}")

if __name__ == "__main__":
    main()
