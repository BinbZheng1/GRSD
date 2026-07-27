# GRSD

GRSD (Group-Relative Self-Distillation) is a multi-turn agent reinforcement-learning trainer built on the `verl` runtime. It uses success/failure groups to construct a contrastive privileged prior and modulates the task advantage with a bidirectional teacher-student signal.

This repository contains two GRSD variants:

- **GRSD** (`verl.trainer.main_grsd`): an external OpenAI-compatible service creates trajectory reflections and the contrastive prior.
- **GRSD-Reflect** (`verl.trainer.main_grsd_reflect`): the policy creates reflections and the prior; the external service is used only as an optional rubric judge.

The implementation supports multi-turn ALFWorld and Search-R1-style Search QA. The `verl` and `agent_system` directories provide the distributed training and environment runtime required by both variants.

## Layout

```text
verl/trainer/main_grsd.py                 GRSD entry point
verl/trainer/main_grsd_reflect.py        GRSD-Reflect entry point
verl/trainer/ppo/grsd_utils.py            reflection and turn-level GRSD logic
verl/trainer/ppo/grsd_ray_trainer.py      GRSD trainer
verl/trainer/ppo/grsd_reflect_*.py        policy-native reflection path
examples/run_grsd_*.sh                    ALFWorld/Search launchers
skills/alfworld, skills/search             task skill sources
tests/trainer/ppo/test_grsd_advantage.py  focused GRSD tests
```

## Install

Use Python 3.10-3.12 with a CUDA-enabled PyTorch installation, then install the package and the rollout engine:

```bash
pip install -e .
pip install 'vllm>=0.8.5,<=0.11.0'
```

Install the environment dependencies separately when needed:

```bash
pip install gym==0.26.2 gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld
alfworld-download -f
```

For Search QA, install the local retriever dependencies described by `examples/search/retriever/retrieval_launch.sh` and prepare the Search-R1 dataset with `examples/data_preprocess/preprocess_search_r1_dataset.py`.

## Run

Launchers use paths relative to the repository by default and accept environment overrides for model, data, GPU, GRSD hyperparameters, and API settings. Set an API key explicitly; no credential is stored in this repository.

```bash
export JUDGE_API_KEY=your_openai_compatible_key
bash examples/run_grsd_alfworld_qwen3_local.sh
```

The ALFWorld wrapper defaults to **GRSD-Reflect**. To run the original external-reflection ablation:

```bash
GRSD_VARIANT=original bash examples/run_grsd_alfworld_qwen3_local.sh
```

For Search QA, start the retrieval service first and then run:

```bash
bash examples/search/retriever/retrieval_launch.sh
bash examples/run_grsd_search_qwen3_local.sh
```

The external service endpoint is configured with `JUDGE_API_BASE` and `JUDGE_MODEL`; the defaults target the public OpenAI-compatible API and can be replaced with any compatible deployment.

## Verify

The focused tests exercise the exponential GRSD signal, invalid-group gating, token/turn modulation, and lambda scheduling:

```bash
python -m unittest tests.trainer.ppo.test_grsd_advantage
```

The maintained public training surface in this repository is GRSD; the
underlying distributed runtime remains in `verl` and `agent_system`.
