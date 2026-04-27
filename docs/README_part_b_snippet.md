## Part B: TWITTER train/validation/test split manifest

Part B creates a fixed TWITTER split manifest from the Part A `graphs_index.csv`.
This manifest is committed to the repository because EGN/HGS training and
evaluation should use the exact same graph split across runs.

```bash
python -m gco_hpif.cli.make_twitter_split \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --output data/manifests/twitter_split_60_20_20.csv
```

For the current Part A TWITTER manifest with 973 graphs, the default rounded
60/20/remainder split is:

- `train`: 584 graphs
- `validation`: 195 graphs
- `test`: 194 graphs

The generated CSV contains the ordered `graph_id` values and the assigned split.

To inspect the manifest:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("data/manifests/twitter_split_60_20_20.csv")
print(df.groupby("split").size())
print(df.head())
print(df.tail())
PY
```

If you need to exactly reproduce a previously used split count, pass explicit
counts:

```bash
python -m gco_hpif.cli.make_twitter_split \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --output data/manifests/twitter_split_60_20_20.csv \
  --train-count 584 \
  --validation-count 195
```
