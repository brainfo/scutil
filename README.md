## Installation

```{bash}
pip install scutil
```

This is for scRNAseq data analysis where

## usage

for qc and low dim visualization, starting from h5ad file,

as in the tests/

```{py}
import scutil as su
import scanpy as sc
import json
su.check_workdir("../")
adata = sc.read_h5ad("tests/_data/test.h5ad")

with open('tests/config/params.json', 'r') as f:
    config = json.load(f)

name = config['project_name']
adata = sc.read_h5ad(f"tests/_data/{name}.h5ad")
su.filter_adata(adata, **config['filter_params'])
su.norm_hvg(adata, name, n_top_genes=1000)
su.pca(adata, name, 30, pearson=False)
su.tsne_and_umap(adata, name, n_comps=10, pearson=False, key='celltype')

su.write_adata(adata, f'tests/_data/{name}_qc_vis.h5ad')
```
