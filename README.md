## Installation

```{bash}
pip install scutil
```

This is for scRNAseq data analysis where

## usage

- starting with cellranger output matrices from multiple samples:

```{py}
import scutil as su
import scanpy as sc
import json
su.check_workdir("../")

with open('tests/config/params.json', 'r') as f:
    config = json.load(f)

name = config['project_name']

raw = su.anndata_from_matrix(config['matdir'])

doublet_dict = defaultdict()
for sample_name in raw.obs['sample'].unique():
    sample = raw[raw.obs['sample'] == sample_name].copy()
    sc.external.pp.scrublet(sample, random_state=0, threshold=0.1)
    su.doublet_plot("tests/", sample_name, sample)
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
ad_all.write_h5ad(f"tests/_data/output/{name}_filter.h5ad", compression='gzip')
```

- for qc and low dim visualization, starting from preprocessed h5ad file, as from the previous step ⬆,

```{py}
import scutil as su
import scanpy as sc
import json
su.check_workdir("../")
adata = sc.read_h5ad("tests/_data/test.h5ad")

with open('tests/config/params.json', 'r') as f:
    config = json.load(f)

name = config['project_name']
adata = sc.read_h5ad(f"tests/_data/{name}_filter.h5ad")
su.filter_adata(adata, **config['filter_params'])
su.norm_hvg(adata, name, n_top_genes=1000)
su.pca(adata, name, 30, pearson=False)
su.tsne_and_umap(adata, name, n_comps=10, pearson=False, key='celltype')

su.write_adata(adata, f'tests/_data/{name}_qc_vis.h5ad')
```
