
This is for scRNAseq data analysis where

## gene level analysis

- Plot the bar output from gene ontology enrichment analysis
- Network analysis given any edge list

## scanpy support

utils for handy using scanpy

The zarr version should be below 3. Likely 2.18.2.

usage:

```{py}
import scanpy as sc
import os
import matplotlib.pyplot as plt
import sc_utils.scanpy_utils as scu

plt.style.use('misc/general.mplstyle')
# project_name =
# workdir = '/data/run01/scv6813/XGF/eyeball'
# matdir = f'{workdir}/matrices'
# os.chdir(workdir)

raw = scu.anndata_from_matrix(matdir)
for sample_name in raw.obs['sample'].unique():
    sample = raw[raw.obs['sample'] == sample_name]
    sc.external.pp.scrublet(sample)
    scu.doublet_plot(workdir, sample_name, sample)
```
