from dataclasses import dataclass, replace
from typing import Optional, Tuple, List
from transformers import PreTrainedModel, PreTrainedTokenizerBase, LogitsProcessorList, NoBadWordsLogitsProcessor
from rrh_logits_processor import RRHConfig, RRHProcessor


@dataclass
class RecurrenceConfig:
    steps: int = 0               # rollout depth (0 = baseline)
    entropy_gate: float = 1.3    # gate for RRH triggering
    max_new_tokens: int = 400
    temperature: float = 0.0
    # RRH/search knobs
    top_k: int = 2               # max branch width
    time_limit_s: float = 12.0   # wallclock cap per generation
    max_branches: int = 4        # total RRH firings per sequence
    margin_gate: float = 1.0     # skip if top1-top2 margin is large
    only_answer_phase: bool = True
    ban_meta_placeholders: bool = True
    # RRH residual parameters
    rrh_alpha: float = 0.75
    rrh_tau: float = 0.5
    rrh_clip: float = 2.0
    rrh_token_time_ms: float = 35.0
    rrh_cooloff_tokens: int = 24
    rrh_min_improve_eps: float = 1e-4


    # (Old adaptive lookahead removed; RRH path below is the only recurrence used.)


@torch.no_grad()
def generate_with_recurrence(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    cfg: RecurrenceConfig,
    device: Optional[str] = None,
) -> str:
    if device is None:
        device = next(model.parameters()).device
    # Baseline path if no recurrence or sampling is enabled
    if cfg.steps <= 0 or (cfg.temperature and cfg.temperature > 0.0):
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        do_sample = cfg.temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": cfg.max_new_tokens,
            "use_cache": True,
            "pad_token_id": tokenizer.eos_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = cfg.temperature
        out_ids = model.generate(**enc, **gen_kwargs)
        return tokenizer.decode(out_ids[0], skip_special_tokens=True)

    # RRH logits processor (residual head lookahead)
    ar = RRHConfig(
        entropy_hi=(cfg.entropy_gate if cfg.entropy_gate > 0 else 1.3),
        margin_lo=getattr(cfg, "margin_gate", 0.25),
        topk=getattr(cfg, "top_k", 2),
        depth=max(1, cfg.steps),
        alpha=cfg.rrh_alpha,
        tau=cfg.rrh_tau,
        clip=cfg.rrh_clip,
        token_time_ms=cfg.rrh_token_time_ms,
        cooloff_tokens=cfg.rrh_cooloff_tokens,
        min_improve_eps=cfg.rrh_min_improve_eps,
        eos_id=tokenizer.eos_token_id,
        max_time_s=getattr(cfg, "time_limit_s", 12.0),
        only_answer_phase=getattr(cfg, "only_answer_phase", True),
        max_branches=getattr(cfg, "max_branches", 4),
    )
    proc_list = []
    if getattr(cfg, "ban_meta_placeholders", True):
        phrases = ["<yes/no>", "<number>", "<a/b>", "<weekday>", "<box>"]
        bad_ids = [tokenizer.encode(p, add_special_tokens=False) for p in phrases]
        bad_ids = [ids for ids in bad_ids if ids]
        if bad_ids:
            # Place bans BEFORE RRH so RRH sees masked logits
            proc_list.append(NoBadWordsLogitsProcessor(bad_ids, eos_token_id=tokenizer.eos_token_id))
    # RRH after bans
    proc_list.append(RRHProcessor(model, tokenizer, ar))
    processors = LogitsProcessorList(proc_list)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out_ids = model.generate(
        **enc,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=False,
        logits_processor=processors,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


def refine_tail_then_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    cfg: RecurrenceConfig,
    device: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """
    Backwards-compatible wrapper expected by older harnesses.
    Allows overriding max_new_tokens/temperature without mutating cfg in-place.
    """
    if max_new_tokens is None and temperature is None:
        return generate_with_recurrence(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            cfg=cfg,
            device=device,
        )

    updated = replace(
        cfg,
        max_new_tokens=(max_new_tokens if max_new_tokens is not None else cfg.max_new_tokens),
        temperature=(temperature if temperature is not None else cfg.temperature),
    )
    return generate_with_recurrence(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        cfg=updated,
        device=device,
    )
