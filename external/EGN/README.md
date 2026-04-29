# External EGN source location

This folder is reserved for a local checkout of the upstream EGN / Erdos Goes Neural code.

Do not commit the upstream EGN source into this repository unless the upstream authors provide an explicit license or written permission.

To run Part D.4 locally:

```bash
mkdir -p external/EGN
git clone https://github.com/Stalence/erdos_neu.git external/EGN/erdos_neu
```

The wrapper expects:

```text
external/EGN/erdos_neu/models.py
external/EGN/erdos_neu/cut_utils.py
external/EGN/erdos_neu/modules_and_utils.py
```

You can also store EGN somewhere else and pass:

```bash
python -m gco_hpif.cli.run_egn_solver --egn-root /path/to/erdos_neu ...
```

or set:

```bash
export GCO_HPIF_EGN_ROOT=/path/to/erdos_neu
```
