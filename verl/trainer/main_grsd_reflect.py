"""
Main entry point for GRSD (Group-Reflective Self-Distillation).

Difference from main_grsd.py:
  * The privileged prior z_x is NOT built by an external API LLM. Instead the
    POLICY ITSELF generates both the per-trajectory reflection s_i (Stage A,
    TRAINABLE) and the contrastive prior z_x (Stage B, stop-grad snapshot).
  * The external API LLM is used ONLY as a rubric judge (ReflectionJudge) that
    scores each reflection on a discrete {0,1,2,3} scale. That score becomes the
    reward for a small alpha-weighted GRPO update on the reflection tokens
    (paper: L = L_task + alpha * L_ref).

The historical ``reflect`` suffix is retained for checkpoint and import
compatibility. ``main_grsd.py`` is the external-reflection ablation.
"""

import hydra
import ray
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_grsd_reflect(config)


def run_grsd_reflect(config) -> None:
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = GRSDReflectTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class GRSDReflectTaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        grsd_cfg = config.algorithm.get("grsd", {})

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from agent_system.environments import make_envs

        envs, val_envs = make_envs(config)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == "episode":
            from agent_system.reward_manager import EpisodeRewardManager

            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError

        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        assert config.actor_rollout_ref.rollout.n == 1, (
            "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. "
            "In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"
        )

        from agent_system.multi_turn_rollout import TrajectoryCollector

        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # SkillProvider kept only for base-class signature compatibility; the
        # policy-native trainer builds z_x online and never reads static skills.
        from verl.trainer.ppo.rlsd_utils import SkillProvider

        skills_dir = grsd_cfg.get("skills_dir", "skills/alfworld")
        skill_all = grsd_cfg.get("skill_all", False)
        skill_provider = SkillProvider(skills_dir=skills_dir, skill_all=skill_all)

        # External LLM used ONLY as a discrete {0,1,2,3} rubric judge for reflections.
        judge_enabled = grsd_cfg.get("judge_enabled", True)
        reflection_judge = None
        if judge_enabled:
            from verl.trainer.ppo.grsd_reflect_utils import ReflectionJudge

            reflection_judge = ReflectionJudge(
                temperature=grsd_cfg.get("judge_temperature", 0.0),
                max_tokens=grsd_cfg.get("judge_max_tokens", 16),
                timeout=grsd_cfg.get("judge_timeout", 60.0),
                max_retries=grsd_cfg.get("judge_max_retries", 2),
                max_turns_in_prompt=grsd_cfg.get("reflect_max_turns", 50),
                max_chars_per_obs=grsd_cfg.get("reflect_max_chars_per_obs", 400),
                max_concurrency=grsd_cfg.get("judge_max_concurrency", 16),
            )
            print(f"[GRSD] ReflectionJudge API base={reflection_judge.api_base} model={reflection_judge.model} "
                  f"max_concurrency={reflection_judge.max_concurrency}")
        else:
            print("[GRSD] ReflectionJudge disabled; external API scoring and reflection GRPO are off")
        print(f"[GRSD] reflect_loss_coef(alpha)={grsd_cfg.get('reflect_loss_coef', 0.01)} "
              f"reflect_do_sample={grsd_cfg.get('reflect_do_sample', True)}")
        print(f"[GRSD] lambda={grsd_cfg.get('grsd_lambda', 0.5)} eta={grsd_cfg.get('eta', 0.0)} "
              f"g_hat_max={grsd_cfg.get('g_hat_max', 3.0)} "
              f"warmdown_steps={grsd_cfg.get('warmdown_steps', -1)}")

        from verl.trainer.ppo.grsd_reflect_ray_trainer import GRSDReflectRayTrainer

        trainer = GRSDReflectRayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
            skill_provider=skill_provider,
            skill_reflector=None,
            reflection_judge=reflection_judge,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
