# External solver: MoMC

This directory contains the third-party MoMC C source used by the Part D.3 solver wrapper.

## Source and attribution

Original source URL: https://home.mis.u-picardie.fr/~cli/MoMC2016.c

The source file includes the following authors/copyright notice:

```text
Copyright <2016> <Chu-Min Li & Hua Jiang>
```

The permissive license text from the source file is preserved in `LICENSE`.

## Build

The GCO-HPIF wrapper auto-compiles the binary when needed. You can also compile manually:

```bash
gcc -O3 -DMOMC src/MOMC2016_1800sec_timeout_aware.c -o bin/MoMC
chmod +x bin/MoMC
```

## Run pattern

The original notebook run pattern is preserved:

```bash
./MoMC graph.clq
```

The wrapper generates DIMACS `.clq` files under the selected solver output directory, runs MoMC, and writes standardized solver outputs:

```text
solver_runs.csv
solver_errors.csv
run_summary.csv
```
