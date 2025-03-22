
## Installation

```{bash}
pip install snRNAtools
```

This is for scRNAseq data analysis where

## scanpy support

utils for handy using scanpy

The zarr version should be below 3. Likely 2.18.2.

usage:

```{py}
import scanpy as sc
import os
import matplotlib.pyplot as plt
import sc_utils.scanpy_utils as su

plt.style.use('/mnt/data/hong/reference/general.mplstyle')

project_name = 'eyeball'
workdir = f'/mnt/data/hong/2024/side_projects/SHJ3_{project_name}'
matdir = f'{workdir}/matrices'
os.chdir(workdir)

raw = su.anndata_from_matrix(matdir)

doublet_dict = defaultdict()
for sample_name in raw.obs['sample'].unique():
    print(sample_name)
    sample = raw[raw.obs['sample'] == sample_name].copy()
    sc.external.pp.scrublet(sample, random_state=404, threshold=0.1)
    su.doublet_plot(workdir, sample_name, sample)
    doublet_dict[sample_name] = sample

no_doublet_dict = defaultdict()
filter_dict = defaultdict()
filter_params = {
    'min_counts':400, 'min_genes':200, 'max_genes' : 5000, 'percent_mt':5, 'percent':3, 'filter_mt':True
}
for sample_name, sample in doublet_dict.items():
    doublet = np.array(sample.obs['predicted_doublet'], dtype=bool)
    no_doublet_dict[sample_name] = sample[~doublet]
for sample_name, sample in no_doublet_dict.items():
    su.qc(sample, f'{sample_name}_no_doublet', workdir, flags={"mt": r"^MT-", "ribo": r"^RP[LS]", "hb": r"^HB"}, order=None, batch_key=None)
    filter_dict[sample_name] = su.filter_adata(sample, **filter_params)
ad_all = ad.concat(list(filter_dict.values()), label='sample', keys=list(filter_dict.keys()), join='outer', index_unique='-', merge='same')
ad_all.write_h5ad(f"{workdir}/output/filter.h5ad", compression='gzip')
su.qc(ad_all, f'{project_name}_clean', workdir, flags={"mt": r"^MT-", "ribo": r"^RP[LS]", "hb": r"^HB"}, order=None, batch_key='sample')
```
