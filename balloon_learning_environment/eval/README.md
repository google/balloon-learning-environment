# Balloon Learning Environment Evaluation
This directory includes scripts for evaluating trained agents in the
Balloon Learning Environment.

## 1. Run Eval
Run the eval script with your agent (see `eval.py`).
For example:

```
python balloon_learning_environment:eval --output_dir=/tmp/ble --agent=random
```
An evaluation suite can be split into shards (if you want to parallelize
the work) using the `--num_shards` and `--shard_idx` flags.


## 2. Combine Json Files From Shards
If you didn't use shards in step 1, skip to 3. If you used shards, each of
them produced a separate json file. They need to be combined with
`combine_eval_shards.py`. For example:

```
python balloon_learning_environment/utils:combine_eval_shards \
--path=/tmp/ble/eval \
--models=station_seeker --models=random
```



## 3. Visualize Your Results With Colab
Open `balloon_learning_environment/colab/visualize_eval.ipynb` and upload your json file.
