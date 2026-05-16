# Part B — TWITTER split manifest

Part B creates the fixed TWITTER train/validation/test split used by trainable solver integrations.

## Purpose

The goal is to make all trainable experiments reproducible by using a fixed split rather than regenerating random partitions.

## Command

```bash
python -m gco_hpif.cli.make_twitter_split \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --output data/manifests/twitter_split_60_20_20.csv
```

## Expected split counts

| Split | Count |
|---|---:|
| train | 584 |
| validation | 195 |
| test | 194 |

## Main output

```text
data/manifests/twitter_split_60_20_20.csv
```

## Notes

This split manifest should be committed because it defines the reproducible TWITTER split used by downstream stages.
