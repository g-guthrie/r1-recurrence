from dataclasses import dataclass
from typing import Optional, List
import os
import time
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, PreTrainedModel, PreTrainedTokenizerBase


@torch.no_grad()
def _entropy1(logits: torch.Tensor) -> float:
    p = F.softmax(logits.float(), dim=-1)
    return float(-(p * (p.clamp_min(1e-9)).log()).sum(dim=-1).item())


@torch.no_grad()
def _argmax_margin(logits: torch.Tensor) -> float:
    vals, _ = torch.topk(logits.float(), k=min(2, logits.size(-1)), dim=-1)
    if vals.size(1) < 2:
        return 1e9
    return float((vals[0, 0] - vals[0, 1]).item())


@torch.no_grad()
def _forward_step(model: PreTrainedModel, token_id: torch.LongTensor, past_kv=None):
    out = model(input_ids=token_id, use_cache=True, past_key_values=past_kv)
    return out.logits[:, -1, :], out.past_key_values


@torch.no_grad()
def _clone_past(past):
    if past is None:
        return None
    if hasattr(past, "copy"):
        return past.copy()
    if hasattr(past, "clone"):
        return past.clone()
    if isinstance(past, (tuple, list)):
        layers = []
        for layer in past:
            if isinstance(layer, (tuple, list)):
                layers.append(tuple(t.clone() for t in layer))
            else:
                layers.append(layer.clone())
        return tuple(layers)
    return past


@dataclass
class RRHConfig:
    entropy_hi: float = 1.3
    margin_lo: float = 0.25
    topk: int = 3                 # max K; dynamic chooser clamps to this
    depth: int = 4                # max D; dynamic chooser clamps to this
    token_time_ms: float = 35.0   # per-token budget (ms)
    cooloff_tokens: int = 24
    alpha: float = 0.75           # residual strength
    tau: float = 0.5              # weight temperature
    clip: float = 2.0             # residual clamp
    min_improve_eps: float = 1e-4
    eos_id: Optional[int] = None
    max_time_s: float = 12.0      # global time guard
    only_answer_phase: bool = True
    max_branches: int = 4         # cap total firings


class RRHProcessor(LogitsProcessor):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, cfg: RRHConfig):
        super().__init__()
        self.model = model
        self.tok = tokenizer
        self.cfg = cfg
        if self.cfg.eos_id is None:
            self.cfg.eos_id = tokenizer.eos_token_id
        self.device = next(model.parameters()).device

        # Shadow cache state
        self._past = None
        self._shadow_len = 0
        self._skip_until = 0
        self._tok_idx = 0
        self._t0 = time.perf_counter()
        self._branches_used = 0
        self._last_advanced_id: Optional[int] = None
        self._trace = os.getenv("RRH_TRACE", "0") not in ("", "0", "false", "False")

    @torch.no_grad()
    def _sync_cache(self, input_ids: torch.LongTensor):
        seq_len = int(input_ids.size(1))
        if self._past is None:
            out = self.model(input_ids=input_ids.to(self.device), use_cache=True)
            self._past = out.past_key_values
            self._shadow_len = seq_len
            return
        if self._shadow_len == seq_len:
            return
        tail = input_ids[:, self._shadow_len:seq_len].to(self.device)
        for i in range(tail.size(1)):
            _, self._past = _forward_step(self.model, tail[:, i:i+1], self._past)
        self._shadow_len = seq_len

    @torch.no_grad()
    def _in_answer_phase(self, input_ids: torch.LongTensor) -> bool:
        if input_ids.size(1) == 0:
            return False
        tail_len = min(120, int(input_ids.size(1)))
        tail = input_ids[:, -tail_len:]
        try:
            txt = self.tok.decode(tail[0], skip_special_tokens=True).lower()
        except Exception:
            return False
        if ("final answer" in txt) and txt.rstrip().endswith(":"):
            return True
        for tag in ("<yes/no>", "<number>", "<a/b>", "<weekday>", "<box>"):
            if tag in txt:
                return True
        return False

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Batch>1: no-op
        if input_ids.size(0) != 1:
            self._tok_idx += 1
            return scores

        # Global guard
        if (time.perf_counter() - self._t0) > self.cfg.max_time_s:
            self._tok_idx += 1
            return scores

        # Desync guard
        if self._last_advanced_id is not None and self._shadow_len == int(input_ids.size(1)):
            actual = int(input_ids[0, -1].item())
            if actual != self._last_advanced_id:
                self._past = None
                self._shadow_len = 0
                if self._trace:
                    print(f"[RRH] Desync; rebuild (actual={actual}, ours={self._last_advanced_id})")
        self._last_advanced_id = None

        # Align cache with prefix
        self._sync_cache(input_ids)

        logits = scores
        ent = _entropy1(logits)
        margin = _argmax_margin(logits)

        gated = (ent >= self.cfg.entropy_hi) and (margin <= self.cfg.margin_lo)
        if self.cfg.only_answer_phase and not self._in_answer_phase(input_ids):
            gated = False
        if (self._tok_idx < self._skip_until) or (self.cfg.depth <= 0) or (self.cfg.topk <= 1):
            gated = False
        if self._branches_used >= self.cfg.max_branches:
            gated = False

        if not gated:
            next_id = int(logits.argmax(dim=-1, keepdim=True).item())
            _, self._past = _forward_step(self.model, torch.tensor([[next_id]], device=self.device), self._past)
            self._shadow_len += 1
            self._last_advanced_id = next_id
            self._tok_idx += 1
            return scores

        self._branches_used += 1
        per_token_deadline = time.perf_counter() + (self.cfg.token_time_ms / 1000.0)

        # Dynamic K,D
        if margin <= 0.10:
            dyn_topk = 4
        elif margin <= 0.20:
            dyn_topk = 3
        elif margin <= 0.35:
            dyn_topk = 2
        else:
            dyn_topk = 1
        if ent >= 1.8:
            dyn_depth = 6
        elif ent >= 1.4:
            dyn_depth = 4
        else:
            dyn_depth = 2
        K = int(max(1, min(self.cfg.topk, dyn_topk, logits.size(-1))))
        D = int(max(1, min(self.cfg.depth, dyn_depth)))

        _, topk_ids = torch.topk(logits, k=K, dim=-1)
        base_lp = torch.log_softmax(logits, dim=-1)

        cand_first_logits: List[torch.Tensor] = []
        cand_first_past: List = []
        cand_scores: List[float] = []

        for j in range(K):
            if time.perf_counter() > per_token_deadline:
                break
            cand = topk_ids[:, j:j + 1]
            first_logits, first_past = _forward_step(self.model, cand, _clone_past(self._past))

            steps = 1
            plog = float(base_lp[0, cand.item()].item())
            ent_sum = _entropy1(first_logits)
            rl_logits, rl_past = first_logits, first_past
            no_progress = 0
            while steps < D:
                if time.perf_counter() > per_token_deadline:
                    break
                nxt = rl_logits.argmax(dim=-1, keepdim=True)
                plog += float(torch.log_softmax(rl_logits, dim=-1)[0, nxt.item()].item())
                if steps > 1 and int(nxt.item()) == int(cand.item()):
                    no_progress += 1
                rl_logits, rl_past = _forward_step(self.model, nxt, rl_past)
                ent_sum += _entropy1(rl_logits)
                steps += 1
                if int(nxt.item()) == self.cfg.eos_id:
                    break

            avg_lp = plog / steps
            avg_ent = ent_sum / steps
            score = avg_lp + 0.7 * max(0.0, ent - avg_ent) - 0.02 * no_progress

            cand_first_logits.append(first_logits)
            cand_first_past.append(first_past)
            cand_scores.append(score)

        if not cand_scores:
            next_id = int(logits.argmax(dim=-1, keepdim=True).item())
            _, self._past = _forward_step(self.model, torch.tensor([[next_id]], device=self.device), self._past)
            self._shadow_len += 1
            self._last_advanced_id = next_id
            self._tok_idx += 1
            return scores

        w = torch.tensor(cand_scores, device=logits.device, dtype=logits.dtype)
        w = torch.softmax((w - w.max()) / self.cfg.tau, dim=0)
        L_blend = torch.stack([L.squeeze(0) for L in cand_first_logits], dim=0)
        L_blend = (w[:, None] * L_blend).sum(dim=0, keepdim=True)

        delta = (L_blend - logits).clamp(-self.cfg.clip, self.cfg.clip)
        scale = self.cfg.alpha * (1.0 / (1.0 + margin))
        adjusted = logits + scale * delta

        greedy_id = int(logits.argmax(dim=-1, keepdim=True).item())
        rrh_id = int(adjusted.argmax(dim=-1, keepdim=True).item())
        if (rrh_id == greedy_id) and (self.cfg.min_improve_eps > 0.0):
            _, self._past = _forward_step(self.model, torch.tensor([[greedy_id]], device=self.device), self._past)
            self._shadow_len += 1
            self._last_advanced_id = greedy_id
            self._tok_idx += 1
            return scores

        _, self._past = _forward_step(self.model, torch.tensor([[rrh_id]], device=self.device), self._past)
        self._shadow_len += 1
        self._last_advanced_id = rrh_id
        self._tok_idx += 1

        if time.perf_counter() > per_token_deadline:
            self._skip_until = self._tok_idx + self.cfg.cooloff_tokens

        if self._trace:
            print(f"[RRH] fired: ent={ent:.2f} margin={margin:.2f} K={K} D={D} choice={rrh_id} greedy={greedy_id}")

        return adjusted

