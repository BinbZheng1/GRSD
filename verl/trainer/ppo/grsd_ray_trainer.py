"""Trainer for the external-reflection GRSD ablation.

Extends RLSDRayTrainer. Like RLSD it MODULATES the GRPO advantage with the
teacher-student signal (it does NOT add an auxiliary loss). Unlike RLSD:
  * modulation is TURN-LEVEL (per flattened turn row), not token-level;
  * modulation is BIDIRECTIONAL: q>0 amplifies, q<0 attenuates;
  * the teacher's privileged information z_x is built ONLINE per GRPO group
    by an external LLM (SkillReflector), via a two-stage contrastive process:
      Stage A: reflect each trajectory (success -> what to do; fail -> avoid);
      Stage B: induce a contrastive prior z_x, only when the group has BOTH a
               success and a failure (validity gate m_x).

The student log-probs reuse the framework's recomputed ``old_log_probs``
(the FSDP actor forward already done in fit); the teacher needs one extra
forward with z_x prepended. The final modulated advantage is written back to
``batch.batch["advantages"]`` (shape unchanged), so the actor needs no change.
"""

from pprint import pprint

import numpy as np
import ray
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import (
    _timer,
    apply_invalid_action_penalty,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.rlsd_utils import SkillProvider
from verl.trainer.ppo.rlsd_ray_trainer import RLSDRayTrainer, build_teacher_batch
from verl.trainer.ppo.grsd_utils import SkillReflector, compute_grsd_turn_advantage
from verl.utils.metric import reduce_metrics
from verl.utils.torch_functional import masked_mean
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)

from agent_system.multi_turn_rollout import adjust_batch


def build_teacher_batch_with_text(
    batch: DataProto,
    row_skill_text,
    tokenizer,
    max_prompt_length: int,
    truncation: str = "left",
):
    """Build a teacher batch by prepending a PER-ROW privileged prior z_x.

    Mirrors rlsd_ray_trainer.build_teacher_batch, but the skill text is given
    explicitly per row (online prior) instead of looked up from a static
    SkillProvider. Rows with empty text get no prefix.
    """
    from verl.utils.model import compute_position_id_with_mask

    bs = batch.batch["input_ids"].size(0)
    response_length = batch.batch["responses"].size(1)

    teacher_input_ids_list = []
    teacher_attention_mask_list = []
    teacher_position_ids_list = []

    for i in range(bs):
        original_input_ids = batch.batch["input_ids"][i]
        original_attention_mask = batch.batch["attention_mask"][i]
        prompt_length = original_input_ids.size(0) - response_length

        prompt_ids = original_input_ids[:prompt_length]
        prompt_mask = original_attention_mask[:prompt_length]

        valid_start = prompt_mask.nonzero(as_tuple=True)[0]
        valid_start = valid_start[0].item() if len(valid_start) > 0 else 0
        valid_prompt_ids = prompt_ids[valid_start:]
        prompt_text = tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)

        skill_text = row_skill_text[i] if i < len(row_skill_text) else ""
        if skill_text:
            teacher_prompt_text = f"[Privileged Skill Information]\n{skill_text}\n\n" + prompt_text
        else:
            teacher_prompt_text = prompt_text

        teacher_prompt_ids = tokenizer.encode(teacher_prompt_text, add_special_tokens=False)
        if len(teacher_prompt_ids) > max_prompt_length:
            teacher_prompt_ids = teacher_prompt_ids[-max_prompt_length:]

        teacher_prompt_ids = torch.tensor(teacher_prompt_ids, dtype=torch.long)
        actual_prompt_len = len(teacher_prompt_ids)

        pad_length = max_prompt_length - actual_prompt_len
        if pad_length > 0:
            pad_ids = torch.full((pad_length,), tokenizer.pad_token_id, dtype=torch.long)
            teacher_prompt_ids = torch.cat([pad_ids, teacher_prompt_ids])
            t_prompt_mask = torch.cat([
                torch.zeros(pad_length, dtype=torch.long),
                torch.ones(actual_prompt_len, dtype=torch.long),
            ])
        else:
            t_prompt_mask = torch.ones(actual_prompt_len, dtype=torch.long)

        response_ids = batch.batch["responses"][i]
        response_mask = original_attention_mask[-response_length:]

        teacher_full_ids = torch.cat([teacher_prompt_ids, response_ids])
        teacher_full_mask = torch.cat([t_prompt_mask, response_mask])
        teacher_position_ids = compute_position_id_with_mask(teacher_full_mask.unsqueeze(0))[0]

        teacher_input_ids_list.append(teacher_full_ids)
        teacher_attention_mask_list.append(teacher_full_mask)
        teacher_position_ids_list.append(teacher_position_ids)

    teacher_batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.stack(teacher_input_ids_list),
            "attention_mask": torch.stack(teacher_attention_mask_list),
            "position_ids": torch.stack(teacher_position_ids_list),
            "responses": batch.batch["responses"],
        },
    )
    return teacher_batch


class GRSDRayTrainer(RLSDRayTrainer):
    """GRSD trainer: online contrastive prior + bidirectional turn-level advantage modulation."""

    def __init__(self, *args, skill_provider: SkillProvider = None, skill_reflector: SkillReflector = None, **kwargs):
        super().__init__(*args, skill_provider=skill_provider, **kwargs)
        self.skill_reflector = skill_reflector
        grsd_cfg = self.config.algorithm.get("grsd", {})
        self.grsd_lambda_init = grsd_cfg.get("grsd_lambda", 0.5)
        self.grsd_eta = grsd_cfg.get("eta", 0.0)
        self.grsd_g_hat_max = grsd_cfg.get("g_hat_max", 3.0)
        self.grsd_warmdown_steps = grsd_cfg.get("warmdown_steps", -1)
        # The teacher prompt contains the task prompt plus a compact privileged
        # prior, so it needs its own budget instead of sharing the task limit.
        self.grsd_teacher_max_prompt_length = int(
            grsd_cfg.get("teacher_max_prompt_length", self.config.data.max_prompt_length)
        )
        if self.grsd_teacher_max_prompt_length < self.config.data.max_prompt_length:
            raise ValueError(
                "algorithm.grsd.teacher_max_prompt_length must be at least "
                f"data.max_prompt_length ({self.config.data.max_prompt_length}), got "
                f"{self.grsd_teacher_max_prompt_length}"
            )

    def _get_grsd_lambda(self, step: int) -> float:
        """Return fixed lambda, or linearly decay it to zero by warmdown_steps."""
        if self.grsd_warmdown_steps <= 0:
            return self.grsd_lambda_init
        if step >= self.grsd_warmdown_steps:
            return 0.0
        return self.grsd_lambda_init * (1.0 - step / self.grsd_warmdown_steps)

    # ------------------------------------------------------------------ #
    # Online contrastive prior construction (Stage A + Stage B)
    # ------------------------------------------------------------------ #
    def _decode_turn(self, batch: DataProto, i: int):
        """Decode (obs, action) text for flattened row i."""
        prompt_ids = batch.batch["prompts"][i]
        resp_ids = batch.batch["responses"][i]
        obs = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        action = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
        return obs, action

    def _build_group_priors(self, batch: DataProto):
        """Build per-group privileged prior z_x via the external LLM.

        Returns:
            row_skill_text: list[str] of length bs (z_x for each row; "" if m_x=0).
            row_valid: np.ndarray bool of length bs (m_x per row).
            metrics: dict.
        """
        bs = batch.batch["responses"].size(0)
        uids = batch.non_tensor_batch["uid"]
        traj_uids = batch.non_tensor_batch["traj_uid"]
        turn_steps = batch.non_tensor_batch.get("turn_step", np.zeros(bs, dtype=object))
        episode_rewards = batch.non_tensor_batch.get("episode_rewards", np.zeros(bs, dtype=object))

        # Group rows by uid -> trajectory -> ordered turns.
        group = {}
        traj_reward = {}  # (uid, traj_uid) -> episode reward
        traj_order = {}  # uid -> list of traj_uid in first-seen order
        for i in range(bs):
            uid = uids[i]
            tj = traj_uids[i]
            obs, action = self._decode_turn(batch, i)
            group.setdefault(uid, {}).setdefault(tj, []).append((int(turn_steps[i]), obs, action))
            traj_reward[(uid, tj)] = float(episode_rewards[i])
            traj_order.setdefault(uid, [])
            if tj not in traj_order[uid]:
                traj_order[uid].append(tj)

        def _task_text(uid, tj):
            turns = sorted(group[uid][tj], key=lambda x: x[0])
            return turns[0][1] if turns else ""

        uid_to_prior = {}  # uid -> z_x text (or None)
        uid_valid = {}  # uid -> bool m_x
        for uid, trajs in group.items():
            pos_tjs = [tj for tj in trajs if traj_reward[(uid, tj)] > 0.0]
            neg_tjs = [tj for tj in trajs if traj_reward[(uid, tj)] <= 0.0]

            # Validity gate m_x: need both a success and a failure.
            if len(pos_tjs) == 0 or len(neg_tjs) == 0:
                uid_to_prior[uid] = None
                uid_valid[uid] = False
                continue

            task_text = _task_text(uid, traj_order[uid][0])

            # Stage A: reflect each trajectory.
            pos_refl, neg_refl = [], []
            ok = True
            for tj in pos_tjs:
                turns_sorted = [
                    {"obs": o, "action": a} for (_, o, a) in sorted(group[uid][tj], key=lambda x: x[0])
                ]
                r = self.skill_reflector.reflect_trajectory(task_text, turns_sorted, success=True)
                if r is None:
                    ok = False
                    break
                pos_refl.append(r)
            if ok:
                for tj in neg_tjs:
                    turns_sorted = [
                        {"obs": o, "action": a} for (_, o, a) in sorted(group[uid][tj], key=lambda x: x[0])
                    ]
                    r = self.skill_reflector.reflect_trajectory(task_text, turns_sorted, success=False)
                    if r is None:
                        ok = False
                        break
                    neg_refl.append(r)

            # Stage B: contrastive prior.
            prior = None
            if ok:
                prior = self.skill_reflector.build_prior(task_text, pos_refl, neg_refl)

            if prior is None:
                uid_to_prior[uid] = None
                uid_valid[uid] = False
            else:
                uid_to_prior[uid] = prior
                uid_valid[uid] = True

        row_skill_text = []
        row_valid = np.zeros(bs, dtype=bool)
        for i in range(bs):
            uid = uids[i]
            prior = uid_to_prior.get(uid)
            if prior is not None and uid_valid.get(uid, False):
                row_skill_text.append(prior)
                row_valid[i] = True
            else:
                row_skill_text.append("")
                row_valid[i] = False

        n_groups = len(group)
        n_valid_groups = sum(1 for v in uid_valid.values() if v)
        metrics = {
            "grsd/n_groups": float(n_groups),
            "grsd/n_valid_groups": float(n_valid_groups),
            "grsd/api_call_count": float(self.skill_reflector.call_count),
            "grsd/api_fail_count": float(self.skill_reflector.fail_count),
        }
        return row_skill_text, row_valid, metrics

    def _compute_teacher_log_probs_with_priors(self, batch: DataProto, row_skill_text):
        """Teacher forward where each row uses its group's online prior z_x."""
        teacher_batch = build_teacher_batch_with_text(
            batch=batch,
            row_skill_text=row_skill_text,
            tokenizer=self.tokenizer,
            max_prompt_length=self.grsd_teacher_max_prompt_length,
            truncation=self.config.data.get("truncation", "left"),
        )
        teacher_output = self.actor_rollout_wg.compute_log_prob(teacher_batch)
        return teacher_output.batch["old_log_probs"]

    # ---- fit ---- #
    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training")
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                        )

                    del batch
                    batch = gen_batch_output

                    batch = adjust_batch(self.config, batch)
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    # ---- GRSD: build online contrastive priors z_x per group ----
                    with _timer("grsd_skill_reflect", timing_raw):
                        row_skill_text, row_valid, reflect_metrics = self._build_group_priors(batch)
                        metrics.update(reflect_metrics)

                    # ---- GRSD: teacher forward with per-row prior ----
                    with _timer("teacher_forward", timing_raw):
                        teacher_log_probs = self._compute_teacher_log_probs_with_priors(batch, row_skill_text)
                        batch.batch["teacher_log_probs"] = teacher_log_probs

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.actor_rollout_ref.actor.get("use_invalid_action_penalty", True):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                            )
                            metrics.update(invalid_metrics)

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )

                        # ---- GRSD: bidirectional turn-level advantage modulation ----
                        seq_advantages = batch.batch["advantages"]
                        student_log_probs = batch.batch["old_log_probs"]
                        teacher_lp = batch.batch["teacher_log_probs"]
                        response_mask = batch.batch["response_mask"]
                        traj_index = batch.non_tensor_batch["traj_uid"]
                        current_lambda = self._get_grsd_lambda(self.global_steps)

                        # Build integer uid_index for group-level normalization.
                        uids = batch.non_tensor_batch["uid"]
                        uid_list = list(dict.fromkeys(uids))  # preserves order, deduplicates
                        uid_to_int = {u: i for i, u in enumerate(uid_list)}
                        uid_index = np.array([uid_to_int[u] for u in uids], dtype=np.int64)

                        token_advantages, grsd_metrics = compute_grsd_turn_advantage(
                            seq_advantages=seq_advantages,
                            student_log_probs=student_log_probs,
                            teacher_log_probs=teacher_lp,
                            response_mask=response_mask,
                            traj_index=traj_index,
                            valid_mask=row_valid,
                            uid_index=uid_index,
                            grsd_lambda=current_lambda,
                            eta=self.grsd_eta,
                            g_hat_max=self.grsd_g_hat_max,
                        )
                        batch.batch["advantages"] = token_advantages
                        grsd_metrics["grsd/lambda"] = current_lambda
                        metrics.update(grsd_metrics)

                        # teacher_log_probs only used here; drop so the actor's
                        # select_keys path is identical to plain GRPO.
                        batch.batch.pop("teacher_log_probs", None)

                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    test_start_step = self.config.trainer.get("test_start_step", 0)
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or (self.global_steps >= test_start_step and self.global_steps % self.config.trainer.test_freq == 0)):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
