# GRSD Environments

The GRSD launchers use two multi-turn environments.

## ALFWorld

```bash
pip install gym==0.26.2 gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld
alfworld-download -f
```

Set `ALFWORLD_DATA` to the downloaded data directory before launching
`examples/run_grsd_alfworld_qwen3_local.sh`.

## Search QA

Install the Search-R1 environment from
`agent_system/environments/env_package/search/third_party`, then start the
retrieval service with `examples/search/retriever/retrieval_launch.sh`.
Prepare the parquet files with
`examples/data_preprocess/preprocess_search_r1_dataset.py` and launch
`examples/run_grsd_search_qwen3_local.sh`.
