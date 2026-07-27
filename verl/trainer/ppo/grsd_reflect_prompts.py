"""
GRSD-Reflect prompts: the SINGLE source of truth for every prompt used by the
policy-native two-stage skill construction and its rubric judge.

Three prompt families live here:

  * Stage A (reflection s_i)  -> REFLECT_SYSTEM / REFLECT_USER_TEMPLATE
        Shown to the POLICY. The policy completes this; its completion tokens are
        the trainable s_i.
  * Stage B (prior / summary z_x) -> PRIOR_SYSTEM / PRIOR_USER_TEMPLATE
        Shown to the POLICY SNAPSHOT (rollout engine -> stop-gradient). Summarizes
        the group's reflections into a contrastive DO/AVOID guideline.
  * Rubric judge (r^ref_i)   -> JUDGE_SYSTEM / JUDGE_USER_TEMPLATE
        Shown to the external API LLM. Grades each reflection on {0, 1, 2, 3}.

The small ``build_*_user_content`` helpers assemble the user-message strings from
these templates and are kept here so the prompts and their formatting stay in one
place. Consumers (``grsd_reflect_utils`` and ``grsd_reflect_ray_trainer``) import
from THIS module.
"""

from typing import List


# --------------------------------------------------------------------------- #
# Stage A: reflection prompt SHOWN TO THE POLICY (p_ref).
# The policy itself completes this; the completion tokens are the trainable s_i.
# --------------------------------------------------------------------------- #

REFLECT_SYSTEM = (
    "You are the same agent that just attempted an interactive task. Reflect on "
    "your own trajectory and distill a short, concrete strategy you can reuse. "
    "Be specific and grounded in the decisions, actions, and evidence observed."
)

REFLECT_USER_TEMPLATE = """Reflect on ONE of your own completed trajectories.

# Task
{task_text}

# Outcome
This trajectory {outcome_phrase}.

# Your interaction history (turn by turn)
{trajectory_text}

# Instructions
Write 3 numbered key points that capture the decisive behavior of THIS trajectory.
- If it SUCCEEDED: state concretely what you did RIGHT (the decisions/steps that caused success).
- If it FAILED: state concretely what went WRONG and which action(s) to AVOID or do differently.
Each key point: one imperative sentence grounded in the actual actions. Output only the numbered list."""


# --------------------------------------------------------------------------- #
# Stage B: prior-synthesis (summary) prompt SHOWN TO THE POLICY SNAPSHOT (p_prior).
# The snapshot summarizes the group's reflections into a contrastive DO/AVOID
# guide z_x. Generated under no-grad (rollout engine) -> stop-gradient.
# --------------------------------------------------------------------------- #

PRIOR_SYSTEM = (
    "You summarize reflections from several trajectories of the SAME task "
    "(some succeeded, some failed) into one compact contrastive guideline."
)

PRIOR_USER_TEMPLATE = """Below are reflections from multiple trajectories attempting the SAME task.

# Task
{task_text}

# Reflections from SUCCESSFUL trajectories
{positive_block}

# Reflections from FAILED trajectories
{negative_block}

# Instructions
Induce a compact guideline with EXACTLY these two sections and nothing else:

### Key points to DO (from successful trajectories)
- 3-5 imperative bullet points describing the correct decisions/steps that lead to success.

### Mistakes to AVOID (from failed trajectories)
- 3-5 imperative bullet points describing the wrong actions to avoid, derived from the failures.

Be concrete and task-specific. Output only the two sections with their bullet points."""


# --------------------------------------------------------------------------- #
# Rubric judge (external API LLM): scores reflection quality on {0, 1, 2, 3}.
# --------------------------------------------------------------------------- #

JUDGE_SYSTEM = (
    "You are a strict, discriminating rubric grader for self-reflections written "
    "by an interactive task agent. You are given the task, the "
    "verified outcome, the full trajectory, and the agent's reflection. Grade ONLY "
    "the quality of the reflection as a causally-correct, outcome-consistent, and "
    "reusable skill -- NOT whether the trajectory itself succeeded. Be demanding: "
    "Reserve the top score for genuinely flawless reflections, and whenever you"
    "hesitate between two scores, choose the LOWER one."
)

JUDGE_USER_TEMPLATE = """Grade the agent's self-reflection using the rubric below.

# Task
{task_text}

# Verified outcome
This trajectory {outcome_phrase}.

# Trajectory (turn by turn)
{trajectory_text}

# Agent's reflection to grade
{reflection_text}

# What to judge (assess the reflection on ALL four dimensions)
1. Causal correctness -- it pinpoints the DECISIVE turn(s)/action(s) that actually
   caused the outcome, not incidental or cosmetic steps.
2. Outcome consistency -- for a SUCCESS it states what to DO (the winning decisions);
   for a FAILURE it states what went WRONG and what to AVOID/fix. It must never
   contradict the verified outcome.
3. Faithfulness -- every action/step it references actually appears in the trajectory;
   it invents or hallucinates nothing.
4. Actionability -- each point is concrete, specific, and reusable as a skill, not a
   vague platitude such as "be careful", "plan ahead", or "explore more".

# Rubric (choose exactly one integer from 0 to 3)
3 = Excellent. Correctly identifies the decisive turn(s) with sound causal
    reasoning, is fully grounded in the actual trajectory, is consistent with the
    verified outcome, and EVERY point is concrete and actionable. No vague, generic,
    or unsupported content anywhere.
2 = Good. Captures the core decisive behavior, is outcome-consistent and grounded,
    but has minor flaws: e.g. one vague/generic point, misses a secondary factor, or
    is slightly imprecise about which turn was actually decisive.
1 = Weak. Only loosely useful: misses the key decisive turn, is mostly generic or
    vague, OR pairs a correct point with an unsupported/hallucinated one, OR is only
    loosely tied to the outcome.
0 = Poor. Incorrect, contradicts the verified outcome, hallucinates steps that did
    not occur, or is non-actionable / off-topic.

# Calibration (read before scoring)
- Be strict and spread your scores. A merely "reasonable but generic" reflection is
  at most 1.
- Do NOT give 3 unless you genuinely cannot find a single flaw.
- When torn between two scores, always pick the LOWER one.

# Output format
Respond with ONLY a single character: 0, 1, 2, or 3. No words, no punctuation."""


# --------------------------------------------------------------------------- #
# User-message builders (assemble the templates above).
# --------------------------------------------------------------------------- #

def build_reflect_user_content(task_text: str, trajectory_text: str, success: bool) -> str:
    """User message content for Stage A reflection (the policy completes this)."""
    outcome_phrase = "SUCCEEDED" if success else "FAILED"
    return REFLECT_USER_TEMPLATE.format(
        task_text=task_text,
        outcome_phrase=outcome_phrase,
        trajectory_text=trajectory_text,
    )


def build_prior_user_content(task_text: str, positive_reflections: List[str], negative_reflections: List[str]) -> str:
    """User message content for Stage B prior synthesis (policy snapshot completes this)."""
    def _block(items: List[str]) -> str:
        if not items:
            return "(none)"
        return "\n\n".join(f"[Trajectory {i + 1}]\n{r}" for i, r in enumerate(items))

    return PRIOR_USER_TEMPLATE.format(
        task_text=task_text,
        positive_block=_block(positive_reflections),
        negative_block=_block(negative_reflections),
    )


def build_judge_user_content(
    task_text: str, trajectory_text: str, reflection_text: str, success: bool
) -> str:
    """User message content for the rubric judge (external LLM grades this)."""
    outcome_phrase = "SUCCEEDED" if success else "FAILED"
    return JUDGE_USER_TEMPLATE.format(
        task_text=str(task_text)[:1500],
        outcome_phrase=outcome_phrase,
        trajectory_text=trajectory_text,
        reflection_text=str(reflection_text)[:2000],
    )
