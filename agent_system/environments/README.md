# GRSD Environments

The GRSD launchers use three multi-turn environments.

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

## WebShop

WebShop requires Python 3.10, JDK 11 or newer, product JSON files, and a
Pyserini/Lucene index. Install its runtime dependencies with:

```bash
pip install -r agent_system/environments/env_package/webshop/webshop/requirements.txt
```

Keep the large assets outside Git under `${WORKSPACE}/data/webshop` (or set
`WEBSHOP_DATA_DIR` and `WEBSHOP_INDEX_DIR`). The GRSD launcher validates the
assets and creates the source-tree links expected by WebShop. Set
`WEBSHOP_JAVA_HOME` if the JDK is not under `${WORKSPACE}/data/webshop/jdk-11`,
then run `examples/run_grsd_webshop_qwen3_local.sh` or its Qwen2.5 counterpart.
