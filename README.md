## Installation

```{bash}
uv pip install git+https://github.com/brainfo/scutils.git
```

This is for scRNAseq data analysis where

## usage

1. for qc and low dim visualization, starting from h5ad file,

    ```python
    import scutil as su
    import scanpy as sc
    import json
    su.check_workdir("../")

    with open('tests/config/params.json', 'r') as f:
    config = json.load(f)

    name = config['project_name']
    adata = sc.read_h5ad(f"tests/_data/{name}.h5ad")
    su.qc(adata, name, config['workdir'],order=['PLA23', 'PLA15', 'PLA16', 'PLA27'], batch_key = 'sample')
    su.filter_adata(adata, **config['filter_params'])
    su.norm_hvg(adata, name, n_top_genes=1000)
    su.pca(adata, name, 30, pearson=False)
    su.tsne_and_umap(adata, name, n_comps=10, pearson=False, key='celltype')

    su.write_adata(adata, f'tests/_data/{name}_qc_vis.h5ad')
    ```

2. for dot plots with z-score (color) while keep count for pct (size)

    y labels can be on left or right side, and the legend will be located accordingly

    ```python
    from scutil.dotplot_utils import custom_deg_dotplot
    genes = ["EGFR", "ERBB2", "IGF1R"]
    custom_deg_dotplot(
    adata,
    genes=genes,
    groupby="group",
    layer="log_norm",
    max_value=10,        # clip absolute z-score to ±10
    y_right=False,  # put gene names on the right
    swap_axes=True,
    figsize= [7.09/6, 6.69/8],
    vmin = -1,
    vmax=1,
    save = "tests/figures/test_dotplot.pdf"
    )
    ```
