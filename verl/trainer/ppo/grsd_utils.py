"""
GRSD (Group-Relative Self-Distillation) utilities.

Two components:
  1. SkillReflector: an external-LLM client that builds privileged skill
     information per GRPO group via a two-stage process:
       Stage A — per-trajectory skill reflection (what each trajectory did).
       Stage B — contrastive skill prior z_x summarizing what SUCCESSFUL
                 trajectories did RIGHT and what FAILED trajectories should
                 AVOID. Only built when the group has both outcomes (m_x=1).
  2. compute_grsd_turn_advantage: turn-level advantage modulation. The
     self-distillation signal modulates the GRPO outcome-verified advantage;
     the training objective stays GRPO (only the advantage is reshaped).

GRSD's contrastive prior and bidirectional modulation are implemented here:
  - Bidirectional exponential signal: expm1(q)>0 amplifies and expm1(q)<0
    attenuates (no gate drop).
  - Normalization: per-GRPO-group mean-abs scaling for stability.
  - Global clip on g_hat (G_HAT_MAX) replaces the old local g_max clip.
  - tau_gap removed (cancels in the normalized ratio).
  - lambda default lowered to 0.5.
  - LLM temperature default 0.0 for deterministic priors.

All teacher / reflection / prior / weight computation is stop-gradient;
only the student policy is optimized. No privileged info at inference.
"""

import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch


# Keeps expm1 finite even for unexpectedly large teacher-student log-ratios.
_EXP_INPUT_CLIP = 20.0


# --------------------------------------------------------------------------- #
# Stage A / Stage B prompt templates
# --------------------------------------------------------------------------- #

_REFLECT_SYSTEM = (
    "You are an expert analyst of interactive agent trajectories. You read one "
    "trajectory and distill concise, actionable key points. Be specific and "
    "grounded in the decisions, actions, and evidence observed."
)

# Stage A: reflect on ONE trajectory. The outcome label is provided so the
# model frames the key points around success or failure.
_REFLECT_USER_TEMPLATE = """Analyze the following single agent trajectory.

# Task
{task_text}

# Outcome
This trajectory {outcome_phrase}.

# Interaction history (turn by turn)
{trajectory_text}

# Instructions
Extract 3-5 numbered key points that characterize this trajectory's behavior.
- If it SUCCEEDED: state concretely what the agent did RIGHT (the decisions/steps that led to success).
- If it FAILED: state concretely what went WRONG and which action(s) should have been AVOIDED or done differently.
Each key point: one sentence, imperative, grounded in the actual actions. No preamble, output only the numbered list."""


# Stage B: contrastive prior built from the per-trajectory reflections of the
# group. Explicitly asks for BOTH the right-to-do and the to-avoid key points.
_PRIOR_SYSTEM = (
    "You are an expert strategy summarizer for interactive agents. You read "
    "reflections from several trajectories of the "
    "SAME task (some succeeded, some failed) and induce a single compact "
    "guideline of privileged hints."
)

_PRIOR_USER_TEMPLATE = """Below are reflections from multiple trajectories attempting the SAME task.

# Task
{task_text}

# Reflections from SUCCESSFUL trajectories
{positive_block}

# Reflections from FAILED trajectories
{negative_block}

# Instructions
Induce a compact privileged guideline with EXACTLY these two sections and nothing else:

### Key points to DO (from successful trajectories)
- 3-5 imperative bullet points describing the correct decisions/steps that lead to success.

### Mistakes to AVOID (from failed trajectories)
- 3-5 imperative bullet points describing the wrong actions to avoid, derived from the failures.

Be concrete and task-specific. Output only the two sections with their bullet points."""


class SkillReflector:
    """External-LLM client for two-stage privileged skill construction.

    Config is read from environment variables, mirroring the launcher:
        JUDGE_API_BASE | LLM_API_BASE  -> base_url
        JUDGE_API_KEY  | LLM_API_KEY   -> api_key
        JUDGE_MODEL    | LLM_MODEL     -> model

    On any API failure (after retries) the group falls back to no prior
    (m_x treated as 0 -> no modulation -> plain GRPO), so training never
    stalls on a flaky endpoint.
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        max_retries: int = 2,
        max_turns_in_prompt: int = 50,
        max_chars_per_obs: int = 1200,
    ):
        self.api_base = api_base or os.environ.get(
            "JUDGE_API_BASE", os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")
        )
        self.api_key = api_key or os.environ.get(
            "JUDGE_API_KEY", os.environ.get("LLM_API_KEY", "")
        )
        self.model = model or os.environ.get(
            "JUDGE_MODEL", os.environ.get("LLM_MODEL", "gpt-4o-mini")
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_turns_in_prompt = max_turns_in_prompt
        self.max_chars_per_obs = max_chars_per_obs

        self._client = None
        self.fail_count = 0
        self.call_count = 0

    # -- lazy client so importing this module never requires the SDK -------- #
    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.api_base, timeout=self.timeout)
        return self._client

    def _chat(self, system: str, user: str) -> Optional[str]:
        """One chat completion with retries. Returns content string or None."""
        client = self._get_client()
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                self.call_count += 1
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = resp.choices[0].message.content
                if content and content.strip():
                    return content.strip()
                # empty content (e.g. reasoning ate the budget): retry once more
                last_err = "empty_content"
            except Exception as e:  # noqa: BLE001 - want broad fallback
                last_err = f"{type(e).__name__}: {str(e)[:200]}"
            if attempt < self.max_retries:
                time.sleep(1.0 * (attempt + 1))
        self.fail_count += 1
        print(f"[GRSD][SkillReflector] API call failed after retries: {last_err}")
        return None

    # ---- trajectory text formatting -------------------------------------- #
    def _truncate(self, text: str, limit: int) -> str:
        if text is None:
            return ""
        text = str(text)
        if len(text) > limit:
            return text[:limit] + " ...[truncated]"
        return text

    def format_trajectory(self, turns: List[Dict]) -> str:
        """Format a trajectory (list of per-turn dicts) into a readable history.

        Each turn dict is expected to carry 'obs' (observation text) and
        'action' (action text). Missing fields are tolerated.
        """
        turns = turns[: self.max_turns_in_prompt]
        lines = []
        for idx, t in enumerate(turns):
            obs = self._truncate(t.get("obs", ""), self.max_chars_per_obs)
            action = self._truncate(t.get("action", ""), 600)
            lines.append(f"[Turn {idx}] Observation: {obs}")
            lines.append(f"[Turn {idx}] Action: {action}")
        return "\n".join(lines)

    # ---- Stage A --------------------------------------------------------- #
    def reflect_trajectory(self, task_text: str, turns: List[Dict], success: bool) -> Optional[str]:
        outcome_phrase = "SUCCEEDED" if success else "FAILED"
        user = _REFLECT_USER_TEMPLATE.format(
            task_text=self._truncate(task_text, 1500),
            outcome_phrase=outcome_phrase,
            trajectory_text=self.format_trajectory(turns),
        )
        return self._chat(_REFLECT_SYSTEM, user)

    # ---- Stage B --------------------------------------------------------- #
    def build_prior(
        self,
        task_text: str,
        positive_reflections: List[str],
        negative_reflections: List[str],
    ) -> Optional[str]:
        """Build contrastive prior z_x. Requires both lists non-empty (m_x=1)."""
        if not positive_reflections or not negative_reflections:
            return None

        def _block(items: List[str]) -> str:
            return "\n\n".join(f"[Trajectory {i + 1}]\n{r}" for i, r in enumerate(items))

        user = _PRIOR_USER_TEMPLATE.format(
            task_text=self._truncate(task_text, 1500),
            positive_block=_block(positive_reflections),
            negative_block=_block(negative_reflections),
        )
        return self._chat(_PRIOR_SYSTEM, user)


# --------------------------------------------------------------------------- #
# Turn-level advantage modulation
# --------------------------------------------------------------------------- #


def _turn_level_logprob(log_probs: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Average per-token log-prob over each row's valid response tokens.

    Args:
        log_probs: (bs, response_length)
        response_mask: (bs, response_length)
    Returns:
        (bs,) turn-level mean log-prob. Rows with no valid token -> 0.
    """
    tok = (log_probs * response_mask).sum(dim=-1)
    cnt = response_mask.sum(dim=-1).clamp(min=1.0)
    return tok / cnt


def compute_grsd_turn_advantage(
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
):
    """Turn-level GRSD advantage modulation (bidirectional).

    For each flattened row (one turn of one trajectory):
        A_i      = row's GRPO advantage scalar (constant over its tokens)
        ell_tea  = mean teacher log-prob over the turn
        ell_stu  = mean student log-prob over the turn
        delta    = ell_tea - ell_stu              (turn-level skill gap)
        q        = sign(A_i) * delta              (alignment-signed log-ratio)
        q_exp    = exp(clamp(q, -20, 20)) - 1     (zero-centered exponential signal)

        q_norm   = q_exp / (mean_g(|q_exp|) + eps) (group-level normalization)
                   mean-abs computed per GRPO group (same uid/prompt) over valid
                   rows (m_x=1) only. Group-local normalization ensures turns from
                   different prompts or task types never affect each other's scale.

        Bidirectional modulation on q_norm:
          q_norm > eta  -> g = q_norm   amplify  (teacher agrees w/ outcome)
          q_norm < -eta -> g = q_norm   attenuate (teacher contradicts outcome)
          |q_norm| <= eta -> g = 0      no modulation

        g_hat    = clip(g * m_x, -G_HAT_MAX, G_HAT_MAX)
        A_hat    = A_i * clamp(1 + lambda * g_hat * m_x, 0.1, inf)
                   (floor at 0.1 prevents advantage sign flip on strong attenuation)

    where m_x (valid_mask) is 1 only for groups that had both a success and a
    failure (contrastive prior available). For m_x=0 rows, A_hat == A_i
    (degrades to plain GRPO).

    All computation is under no_grad. Returns (token_advantages, metrics).

    Args:
        seq_advantages: (bs, response_length) GRPO advantage broadcast per row.
        student_log_probs: (bs, response_length) log pi_theta(y|x).
        teacher_log_probs: (bs, response_length) log pi_theta(y | x, z_x).
        response_mask: (bs, response_length).
        traj_index: (bs,) trajectory id per row (traj_uid).
        valid_mask: (bs,) bool/0-1, m_x per row.
        uid_index: (bs,) integer group id per row (mapped from uid string).
        grsd_lambda: modulation strength (default 0.5).
        eta: dead-zone threshold; |q_norm| <= eta gets zero modulation (default 0.0).
        g_hat_max: global clip bound on the normalized signal (default 3.0).
        epsilon: numerical stability for division.

    Returns:
        token_advantages: (bs, response_length).
        metrics: dict of scalars for logging.
    """
    with torch.no_grad():
        device = seq_advantages.device
        bs = seq_advantages.size(0)

        # Row-level GRPO advantage scalar: pick value at first valid token.
        first_valid = response_mask.int().argmax(dim=-1)  # (bs,)
        row_idx = torch.arange(bs, device=device)
        A_row = seq_advantages[row_idx, first_valid]  # (bs,)

        ell_tea = _turn_level_logprob(teacher_log_probs, response_mask)  # (bs,)
        ell_stu = _turn_level_logprob(student_log_probs, response_mask)  # (bs,)
        delta = ell_tea - ell_stu  # (bs,)

        sign_A = torch.sign(A_row)  # (bs,)
        q = sign_A * delta  # (bs,) alignment-signed log-ratio
        # expm1 preserves the no-signal fixed point (q=0 -> q_exp=0), unlike
        # exp(q), while retaining bidirectional amplification/attenuation.
        q_exp = torch.expm1(
            torch.clamp(q.float(), min=-_EXP_INPUT_CLIP, max=_EXP_INPUT_CLIP)
        )

        m_x = torch.as_tensor(
            np.asarray(valid_mask, dtype=np.float32), dtype=torch.float32, device=device
        )  # (bs,)
        valid_rows = m_x.sum().clamp(min=1.0)

        # Group-level mean-abs normalization of the exponential signal.
        # Each GRPO group (same uid / prompt) normalizes its own turns independently,
        # so turns from different prompts or task types never pollute each other's
        # reference scale. This also aligns with GRPO advantage computation, which
        # is likewise group-local.
        # We divide by the group mean of |q_exp| over valid rows (m_x=1) only.
        # Using mean-abs (rather than std) keeps the denominator strictly positive
        # even when all q values within a group are similar, and preserves the
        # sign/direction of q without centering.
        group_ids = torch.as_tensor(
            np.asarray(uid_index, dtype=np.int64), dtype=torch.long, device=device
        )  # (bs,)
        num_groups = int(group_ids.max().item()) + 1

        # Per-group mean of |q_exp| over valid rows.
        gabs_sum = torch.zeros(num_groups, device=device).scatter_add_(0, group_ids, q_exp.abs() * m_x)
        gcnt = torch.zeros(num_groups, device=device).scatter_add_(0, group_ids, m_x)
        gcnt_safe = gcnt.clamp(min=1.0)
        g_mean_abs = gabs_sum / gcnt_safe  # (num_groups,)

        q_norm = q_exp / (g_mean_abs[group_ids] + epsilon)  # (bs,) scaled by group mean-abs

        # Bidirectional: g > 0 amplifies, g < 0 attenuates.
        # Dead zone: |q_norm| <= eta -> no modulation.
        g = torch.where(q_norm > eta, q_norm,
                        torch.where(q_norm < -eta, q_norm, torch.zeros_like(q_norm)))

        # Zero-out g for invalid groups (no contrastive prior available).
        # traj_index retained in signature for potential future trajectory-level metrics.
        g_eff = g * m_x

        # q_norm is already group-scaled, so apply only the global safety clip.
        g_hat = torch.clamp(g_eff, min=-g_hat_max, max=g_hat_max)

        # Floor at 0.1 prevents advantage sign flip under strong attenuation.
        modulation = torch.clamp(1.0 + grsd_lambda * g_hat * m_x, min=0.1)  # (bs,)
        A_hat_row = A_row * modulation  # (bs,)

        token_advantages = A_hat_row.unsqueeze(-1) * response_mask  # (bs, R)

        # ---- metrics ----
        # q_norm_scale: mean of per-group mean-abs, proxy for signal magnitude.
        g_mean_abs_per_row = g_mean_abs[group_ids]
        q_norm_scale = (g_mean_abs_per_row * m_x).sum() / valid_rows
        metrics = {
            "grsd/skill_gap_mean": delta.mean().item(),
            "grsd/skill_gap_abs_mean": delta.abs().mean().item(),
            "grsd/aligned_gap_mean": q.mean().item(),
            "grsd/exponential_signal_mean": (q_exp * m_x).sum().item() / valid_rows.item(),
            "grsd/exponential_signal_abs_mean": (q_exp.abs() * m_x).sum().item() / valid_rows.item(),
            "grsd/q_norm_mean": (q_norm * m_x).sum().item() / valid_rows.item(),
            "grsd/q_norm_scale": q_norm_scale.item(),
            "grsd/g_mean": (g_eff.sum() / valid_rows).item(),
            "grsd/g_hat_mean": (g_hat * m_x).sum().item() / valid_rows.item(),
            "grsd/valid_group_ratio": (m_x.sum() / bs).item(),
            "grsd/modulation_mean": (modulation * m_x).sum().item() / valid_rows.item(),
            "grsd/gate_amplify_ratio": ((g > eta).float() * m_x).sum().item() / valid_rows.item(),
            "grsd/gate_attenuate_ratio": ((g < -eta).float() * m_x).sum().item() / valid_rows.item(),
        }

    return token_advantages, metrics
