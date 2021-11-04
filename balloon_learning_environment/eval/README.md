# Balloon Learning Environment Evaluation
This directory includes scripts for evaluating trained agents in the
Balloon Learning Environment.

## 1. Run Eval
The following example code will run eval with the random agent on one seed.
For more configurations, see the flags at [eval.py](https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/eval/eval.py).

```
python -m balloon_learning_environment.eval.eval \
  --output_dir=/tmp/ble/eval \
  --agent=random \
  --suite=micro_eval
```
An [evaluation suite](https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/eval/suites.py)
can be split into shards (if you want to parallelize
the work) using the `--num_shards` and `--shard_idx` flags.


## 2. Combine Json Files From Shards
If you didn't use shards in step 1, skip to 3. If you used shards, each of
them produced a separate json file. They need to be combined with
`combine_eval_shards.py`. For example, if you ran eval with both station_seeker
and the random agent:

```
python -m balloon_learning_environment.utils.combine_eval_shards \
--path=/tmp/ble/eval \
--models=station_seeker --models=random
```



## 3. Visualize Your Results With Colab
Open `balloon_learning_environment/colab/visualize_eval.ipynb` and upload your
json file.
