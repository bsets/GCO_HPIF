# MoMC Attribution Notice

This directory contains the MoMC C source used by Part D.3 of the GCO-HPIF pipeline.

Original source URL: https://home.mis.u-picardie.fr/~cli/MoMC2016.c

Authors / copyright notice in the source file:

- Chu-Min Li
- Hua Jiang

The source file states that this is the revised version published in 2019 and gives the conditional compilation commands for SoMC, DoMC, and MoMC. This repository compiles the MoMC variant with:

```bash
gcc -O3 -DMOMC external/MOMC/src/MOMC2016_1800sec_timeout_aware.c -o external/MOMC/bin/MoMC
```

The license text from the source file is included in `external/MOMC/LICENSE` and should be preserved with copies or substantial portions of the software.
