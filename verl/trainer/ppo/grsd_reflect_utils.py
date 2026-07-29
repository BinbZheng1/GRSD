"""
GRSD utilities for policy-native two-stage guidance construction.

The policy being trained performs both stages with its rollout engine:

* Stage A is trainable. The policy reflects on a completed trajectory, and the
  generated tokens receive a GRPO signal from an external scalar rubric judge.
* Stage B is stop-gradient. A policy snapshot summarizes the group's successful
  and failed reflections into a DO/AVOID prior used only as teacher context.

The reflection reward is a discrete score in {0, 1, 2, 3}, group-normalized
within the rollout group.

All judge / prior / normalization computation here is external to autograd; the
only trainable object is the policy that generated s_i. At inference time none of
this machinery is used.
"""

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import torch


# Keeps expm1 finite even for unexpectedly large teacher-student log-ratios.
_EXP_INPUT_CLIP = 20.0

# Prompts live in a single dedicated module; import them from there.
from verl.trainer.ppo.grsd_reflect_prompts import (
    JUDGE_SYSTEM,
    build_judge_user_content,
)


def format_trajectory(turns: List[Dict], max_turns: int = 50, max_chars_per_obs: int = 400) -> str:
    """Format a trajectory (list of {'obs','action'} dicts) into readable text."""
    turns = turns[:max_turns]
    lines = []
    for idx, t in enumerate(turns):
        obs = str(t.get("obs", ""))
        if len(obs) > max_chars_per_obs:
            obs = obs[:max_chars_per_obs] + " ...[truncated]"
        action = str(t.get("action", ""))
        if len(action) > 600:
            action = action[:600] + " ...[truncated]"
        lines.append(f"[Turn {idx}] Observation: {obs}")
        lines.append(f"[Turn {idx}] Action: {action}")
    return "\n".join(lines)


class ReflectionJudge:
    """External-LLM rubric judge returning a discrete score in {0, 1, 2, 3}.

    Config falls back to the environment variables used by the GRSD launcher:
        JUDGE_API_BASE | LLM_API_BASE  -> base_url
        JUDGE_API_KEY  | LLM_API_KEY   -> api_key
        JUDGE_MODEL    | LLM_MODEL     -> model

    On persistent API failure the score is None; callers treat None as
    "exclude this reflection from the loss" (masked out, not counted in Norm_G).
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 16,
        timeout: float = 60.0,
        max_retries: int = 2,
        max_turns_in_prompt: int = 50,
        max_chars_per_obs: int = 400,
        max_concurrency: int = 16,
    ):
        self.api_base = api_base or os.environ.get(
            "JUDGE_API_BASE", os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")
        )
        self.api_key = api_key or os.environ.get("JUDGE_API_KEY", os.environ.get("LLM_API_KEY", ""))
        self.model = model or os.environ.get("JUDGE_MODEL", os.environ.get("LLM_MODEL", "gpt-4o-mini"))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_turns_in_prompt = max_turns_in_prompt
        self.max_chars_per_obs = max_chars_per_obs
        # Number of concurrent judge requests (the API supports concurrency).
        self.max_concurrency = max(1, int(max_concurrency))

        self._client = None
        self.call_count = 0
        self.fail_count = 0
        # score()/score_batch() may run under a ThreadPoolExecutor, so guard the
        # shared counters. The OpenAI client itself is thread-safe.
        self._counter_lock = threading.Lock()

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.api_base, timeout=self.timeout)
        return self._client

    @staticmethod
    def _parse_score(content: str) -> Optional[int]:
        """Extract the first 0/1/2/3 from the model output; None if unparseable."""
        if content is None:
            return None
        m = re.search(r"[0-3]", content)
        if m is None:
            return None
        return int(m.group(0))

    def score(self, task_text: str, turns: List[Dict], reflection_text: str, success: bool) -> Optional[int]:
        """Return rubric score in {0,1,2,3}, or None on failure."""
        if not reflection_text or not reflection_text.strip():
            return 0  # empty reflection is a valid (bad) sample -> lowest score
        trajectory_text = format_trajectory(turns, self.max_turns_in_prompt, self.max_chars_per_obs)
        user = build_judge_user_content(task_text, trajectory_text, reflection_text, success)
        client = self._get_client()
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._counter_lock:
                    self.call_count += 1
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    # The judge only needs to emit a single 0/1/2/3, so disable
                    # hidden reasoning when the service supports this option.
                    # Without this it burns the whole token budget on reasoning and
                    # returns empty content (finish_reason=length) -> unparseable.
                    extra_body={"thinking": {"type": "disabled"}},
                )
                content = resp.choices[0].message.content
                score = self._parse_score(content)
                if score is not None:
                    return score
                last_err = f"unparseable_content={content!r}"
            except Exception as e:  # noqa: BLE001 - want broad fallback
                last_err = f"{type(e).__name__}: {str(e)[:200]}"
            if attempt < self.max_retries:
                time.sleep(1.0 * (attempt + 1))
        with self._counter_lock:
            self.fail_count += 1
        print(f"[GRSD][Judge] scoring failed after retries: {last_err}")
        return None

    def score_batch(self, items: List[Dict]) -> List[Optional[int]]:
        """Score many reflections concurrently, preserving input order.

        Each item is a dict with keys: task_text, turns, reflection_text, success.
        Returns a list of scores (int in {0,1,2,3}) or None (failure) aligned to
        `items`. Concurrency is bounded by self.max_concurrency; each request
        keeps score()'s own retry/backoff. Falls back to serial when there is a
        single item or concurrency is 1.
        """
        n = len(items)
        if n == 0:
            return []
        if n == 1 or self.max_concurrency == 1:
            return [
                self.score(it["task_text"], it["turns"], it["reflection_text"], it["success"])
                for it in items
            ]

        results: List[Optional[int]] = [None] * n
        workers = min(self.max_concurrency, n)

        def _run(idx_item):
            idx, it = idx_item
            return idx, self.score(
                it["task_text"], it["turns"], it["reflection_text"], it["success"]
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for idx, s in pool.map(_run, list(enumerate(items))):
                results[idx] = s
        return results


def compute_group_normalized_advantage(
    rewards: np.ndarray,
    group_ids: np.ndarray,
    valid: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Group-relative normalized advantage Norm_G(r) (paper Eq. group_norm).

    For each group (same uid), advantage = (r - mean) / (std + eps) computed over
    that group's VALID trajectories only. Invalid rows -> 0. Groups with <2 valid
    rows or zero variance yield 0 advantage (no reflection signal), which is the
    correct GRPO behaviour when there is no within-group contrast.

    Args:
        rewards: (n,) float per-trajectory reflection reward (judge score).
        group_ids: (n,) integer group id per trajectory.
        valid: (n,) bool; False rows (e.g. judge failed) are excluded and set to 0.
        epsilon: numerical stability.

    Returns:
        adv: (n,) float group-normalized advantage.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    group_ids = np.asarray(group_ids)
    valid = np.asarray(valid, dtype=bool)
    adv = np.zeros_like(rewards, dtype=np.float64)

    for g in np.unique(group_ids):
        sel = (group_ids == g) & valid
        if sel.sum() < 2:
            continue  # need >=2 valid trajectories for a meaningful contrast
        vals = rewards[sel]
        mean = vals.mean()
        std = vals.std()
        adv[sel] = (vals - mean) / (std + epsilon)
    return adv.astype(np.float32)


def compute_grsd_token_advantage(
    seq_advantages: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    traj_index: np.ndarray,
    valid_mask: np.ndarray,
    uid_index: np.ndarray,
    grsd_lambda: float = 0.5,
    eta: float = 0.0,
    g_hat_max: float = 3.0,
    epsilon: float = 1e-6,
    token_norm: str = "group",
):
    """TOKEN-level GRSD advantage modulation (ablation of the turn-level version).

    This is the token-granular counterpart of
    ``verl.trainer.ppo.grsd_utils.compute_grsd_turn_advantage``. Instead of
    collapsing each turn to a single scalar skill-gap and giving every token in
    that turn the SAME modulation coefficient, here each token gets its OWN
    coefficient derived from its per-token teacher-student gap. Nothing is
    aggregated within a turn.

    Per token (i = row/turn, t = position):
        A_i        = row's GRPO advantage scalar (constant over its tokens)
        delta_t    = log pi_teacher(y_t) - log pi_student(y_t)   (per-token gap)
        q_t        = sign(A_i) * delta_t                         (alignment-signed log-ratio)
        q_exp_t    = exp(clamp(q_t, -20, 20)) - 1                (zero-centered exponential)
        q_norm_t   = q_exp_t / (group_mean_abs(|q_exp|) + eps)  if token_norm == "group"
                     q_exp_t                                     if token_norm == "none"
                     "group" mirrors the turn-level group-local scale (mean-abs
                     over all valid tokens of the same uid). "none" removes the
                     normalization entirely (RLSD-style): scale is then controlled
                     solely by the +-g_hat_max clamp on the exponential signal.
        g_t        = q_norm_t if |q_norm_t| > eta else 0          (dead zone)
        g_hat_t    = clip(g_t * m_x, -g_hat_max, g_hat_max)
        A_hat_t    = A_i * clamp(1 + lambda * g_hat_t * m_x, 0.1, inf)

    where m_x (valid_mask) is 1 only for groups that had both a success and a
    failure. For m_x=0 rows every token keeps A_i (plain GRPO). All computation
    is under no_grad. Returns (token_advantages, metrics) with the SAME metric
    keys as the turn-level function so ablation curves overlay in wandb.

    Args mirror ``compute_grsd_turn_advantage``.
    """
    with torch.no_grad():
        device = seq_advantages.device
        bs = seq_advantages.size(0)

        # Row-level GRPO advantage scalar (value at first valid token).
        first_valid = response_mask.int().argmax(dim=-1)  # (bs,)
        row_idx = torch.arange(bs, device=device)
        A_row = seq_advantages[row_idx, first_valid]  # (bs,)

        # Per-token skill gap (NO turn aggregation).
        delta_t = (teacher_log_probs - student_log_probs)  # (bs, R)

        sign_A = torch.sign(A_row).unsqueeze(-1)  # (bs, 1)
        q_t = sign_A * delta_t  # (bs, R) alignment-signed per-token log-ratio
        # expm1 preserves the no-signal fixed point (q_t=0 -> q_exp_t=0),
        # unlike exp(q_t), while retaining both positive and negative signals.
        q_exp_t = torch.expm1(
            torch.clamp(q_t.float(), min=-_EXP_INPUT_CLIP, max=_EXP_INPUT_CLIP)
        )

        m_x_row = torch.as_tensor(
            np.asarray(valid_mask, dtype=np.float32), dtype=torch.float32, device=device
        )  # (bs,)
        # Valid-token mask: token must be a real response token AND in a valid group.
        m_x_tok = m_x_row.unsqueeze(-1) * response_mask  # (bs, R)
        valid_tok = m_x_tok.sum().clamp(min=1.0)

        # Group-level mean-abs normalization of q_exp over valid TOKENS.
        # Each GRPO group (same uid / prompt) normalizes its own tokens
        # independently, matching the turn-level version's group-local scale.
        # With token_norm == "none" the normalization is skipped entirely
        # (g_mean_abs := 1); scale is then bounded only by the g_hat_max clamp
        # applied to the exponential per-token signal (RLSD-style control).
        group_ids = torch.as_tensor(
            np.asarray(uid_index, dtype=np.int64), dtype=torch.long, device=device
        )  # (bs,)
        num_groups = int(group_ids.max().item()) + 1

        row_abs_sum = (q_exp_t.abs() * m_x_tok).sum(dim=-1)  # (bs,)
        row_cnt = m_x_tok.sum(dim=-1)  # (bs,)
        gabs_sum = torch.zeros(num_groups, device=device).scatter_add_(0, group_ids, row_abs_sum)
        gcnt = torch.zeros(num_groups, device=device).scatter_add_(0, group_ids, row_cnt)
        if token_norm == "none":
            g_mean_abs = torch.ones(num_groups, device=device)  # no normalization
            q_norm_t = q_exp_t
        else:
            g_mean_abs = gabs_sum / gcnt.clamp(min=1.0)  # (num_groups,)
            q_norm_t = q_exp_t / (g_mean_abs[group_ids].unsqueeze(-1) + epsilon)  # (bs, R)

        # Bidirectional: g > 0 amplifies, g < 0 attenuates. Dead zone |q_norm|<=eta.
        g_t = torch.where(q_norm_t > eta, q_norm_t,
                          torch.where(q_norm_t < -eta, q_norm_t, torch.zeros_like(q_norm_t)))
        g_eff_t = g_t * m_x_tok
        g_hat_t = torch.clamp(g_eff_t, min=-g_hat_max, max=g_hat_max)

        # Floor at 0.1 prevents advantage sign flip under strong attenuation.
        modulation_t = torch.clamp(1.0 + grsd_lambda * g_hat_t * m_x_tok, min=0.1)  # (bs, R)
        token_advantages = A_row.unsqueeze(-1) * modulation_t * response_mask  # (bs, R)

        # ---- metrics (same keys as turn-level for overlay) ----
        resp_sum = response_mask.sum().clamp(min=1.0)
        metrics = {
            "grsd/skill_gap_mean": (delta_t * response_mask).sum().item() / resp_sum.item(),
            "grsd/skill_gap_abs_mean": (delta_t.abs() * response_mask).sum().item() / resp_sum.item(),
            "grsd/aligned_gap_mean": (q_t * response_mask).sum().item() / resp_sum.item(),
            "grsd/exponential_signal_mean": (q_exp_t * m_x_tok).sum().item() / valid_tok.item(),
            "grsd/exponential_signal_abs_mean": (q_exp_t.abs() * m_x_tok).sum().item() / valid_tok.item(),
            "grsd/q_norm_mean": (q_norm_t * m_x_tok).sum().item() / valid_tok.item(),
            "grsd/q_norm_scale": (g_mean_abs[group_ids].unsqueeze(-1) * m_x_tok).sum().item() / valid_tok.item(),
            "grsd/g_mean": (g_eff_t).sum().item() / valid_tok.item(),
            "grsd/g_hat_mean": (g_hat_t * m_x_tok).sum().item() / valid_tok.item(),
            "grsd/valid_group_ratio": (m_x_row.sum() / bs).item(),
            "grsd/modulation_mean": (modulation_t * m_x_tok).sum().item() / valid_tok.item(),
            "grsd/gate_amplify_ratio": ((g_t > eta).float() * m_x_tok).sum().item() / valid_tok.item(),
            "grsd/gate_attenuate_ratio": ((g_t < -eta).float() * m_x_tok).sum().item() / valid_tok.item(),
        }

    return token_advantages, metrics
