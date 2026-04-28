# CliSAT executable

This directory contains the third-party CliSAT executable used by Part D.2 of the GCO-HPIF pipeline.

```text
external/CliSAT/bin/CliSAT
```

Original project: <https://github.com/psanse/CliSAT>

CliSAT is released under the Unlicense / public-domain dedication. See:

```text
external/CliSAT/LICENSE
external/CliSAT/NOTICE.md
```

The wrapper invokes the binary as:

```bash
external/CliSAT/bin/CliSAT <graph_path.clq> <time_limit_seconds> <threads>
```

For the full GCO-HPIF CliSAT run, this is equivalent to the notebook command pattern:

```bash
external/CliSAT/bin/CliSAT <graph_path.clq> 1800 1
```

The included binary is an ELF 64-bit Linux x86-64 executable. On Linux, if it loses executable permissions after download, unzip, or copy, run:

```bash
chmod +x external/CliSAT/bin/CliSAT
```

If you need to run this project on macOS, Windows, or a different Linux architecture, replace the binary with a compatible build and keep the same path, or pass a custom executable path to the CLI using `--clisat-executable`.
