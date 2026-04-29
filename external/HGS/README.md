# External HGS dependency

This directory is intentionally used as a local-only location for the upstream HGS repository:

https://github.com/yimengmin/GeometricScatteringMaximalClique

Licensing note: the upstream repository does not appear to provide a clear open-source license. Therefore, GCO-HPIF does not vendor or redistribute the upstream HGS source code. Clone it locally when running HGS experiments, and keep the cloned source ignored by Git.

Recommended local setup from the GCO-HPIF repository root:

```bash
mkdir -p external/HGS
git clone https://github.com/yimengmin/GeometricScatteringMaximalClique.git   external/HGS/GeometricScatteringMaximalClique
```
