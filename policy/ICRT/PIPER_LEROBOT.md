# Piper LeRobot v3 training

This path trains ICRT from scratch on the black-block Piper demonstrations. It
loads only the released CrossMAE vision encoder; no released ICRT policy
checkpoint is used. The main transformer is configured as a four-layer model
in `config/model_config/custom_transformer_4l.json`.

## Pose contract

The recorded Cartesian `action`, `observation.eef_pose`, and
`observation.leader_eef_pose` are not training inputs. The adapter applies the
same Piper URDF forward kinematics to both robot joint vectors:

- proprioception: FK(`observation.state[:6]`) plus follower gripper
- absolute action target: FK(`action.joint_absolute[:6]`) plus leader gripper

ICRT then converts each future absolute target into a local delta relative to
the current proprioceptive pose.

Task groups are read from `task_index` and `meta/tasks.parquet`. Each episode
must have one task label. Datasets containing several labels are grouped by
label automatically, and ICRT's task barrier prevents a context window from
crossing between labels.

## Temporary stale-tail cleaning

The current recording contains fixed-duration tails after the final fresh
leader command. They are trimmed before 30 Hz to 15 Hz downsampling. The one
call that enables this temporary cleanup is in
`icrt/data/lerobot_v3_adapter.py`:

```python
cleaned_rows = trim_trailing_duplicate_action_timestamps(rows, action_timestamps)
```

For a future dataset that was cleaned during collection, comment out that line
directly. The preceding `cleaned_rows = rows` line already supplies the
unfiltered rows:

```python
cleaned_rows = rows
# cleaned_rows = trim_trailing_duplicate_action_timestamps(rows, action_timestamps)
```

This intentionally is not exposed as a configuration flag.

## Prompt/query construction

An EOS exists only at the final retained frame of an episode. A 512-step
window may begin partway through an episode, so the first EOS can close an
incomplete prefix. ICRT excludes that first EOS from the random split
candidates when later EOS markers exist. The selected later EOS is therefore
an episode boundary: the prefix is prompt context and the next episode starts
the query. With `--model-cfg.policy-cfg.no-prompt-loss`, action loss is applied
only after that boundary.

## Train

Install the local package and launch:

```bash
cd /home/qikang/ICL_codebase/ICL_Policy/policy/ICRT
pip install -e .
bash scripts/train_piper_black_block_4l.sh
```

The script uses batch size 1. AV1 decoding workers can consume substantial CPU
and memory, so adjust `--trainer-cfg.num-workers` for the training machine.

Training is controlled by optimizer-update steps rather than epochs. With
gradient accumulation set to 8, `global_step` advances after eight
micro-batches. The default script trains for 4,000 updates, warms up for 200,
validates every 250, saves every 500, and logs every optimizer update to the
`icrt-piper` W&B project against `global_step`.

The step-based Piper path logs directly to W&B and does not require
TensorBoard. TensorBoard remains an optional compatibility path for the
upstream epoch-based trainer.
