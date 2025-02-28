## these are function tools that supplement scanpy ##
########### or useful when you use scanpy ###########
######### edited on 2022-02-09 by Ruby Jiang ########

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scanpy as sc
import anndata
import re
import seaborn as sns
import matplotlib
import scipy.sparse as sp
from typing import Dict
import logging
import os
import zarr
from anndata.experimental import read_dispatched, write_dispatched, read_elem
from collections import defaultdict

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def anndata_from_h5(file: str,
                    analyzed_barcodes_only: bool = True) -> 'anndata.AnnData':
    """Load an output h5 file into an AnnData object for downstream work.

    Args:
        file: The h5 file
        analyzed_barcodes_only: False to load all barcodes, so that the size of
            the AnnData object will match the size of the input raw count matrix.
            True to load a limited set of barcodes: only those analyzed by the
            algorithm. This allows relevant latent variables to be loaded
            properly into adata.obs and adata.obsm, rather than adata.uns.

    Returns:
        adata: The anndata object, populated with inferred latent variables
            and metadata.

    """

    d = dict_from_h5(file)
    X = sp.csc_matrix((d.pop('data'), d.pop('indices'), d.pop('indptr')),
                      shape=d.pop('shape')).transpose().tocsr()

    # check and see if we have barcode index annotations, and if the file is filtered
    barcode_key = [k for k in d.keys() if (('barcode' in k) and ('ind' in k))]
    if len(barcode_key) > 0:
        max_barcode_ind = d[barcode_key[0]].max()
        filtered_file = (max_barcode_ind >= X.shape[0])
    else:
        filtered_file = True

    if analyzed_barcodes_only:
        if filtered_file:
            # filtered file being read, so we don't need to subset
            print('Assuming we are loading a "filtered" file that contains only cells.')
            pass
        elif 'barcode_indices_for_latents' in d.keys():
            X = X[d['barcode_indices_for_latents'], :]
            d['barcodes'] = d['barcodes'][d['barcode_indices_for_latents']]
        elif 'barcodes_analyzed_inds' in d.keys():
            X = X[d['barcodes_analyzed_inds'], :]
            d['barcodes'] = d['barcodes'][d['barcodes_analyzed_inds']]
        else:
            print('Warning: analyzed_barcodes_only=True, but the key '
                  '"barcodes_analyzed_inds" or "barcode_indices_for_latents" '
                  'is missing from the h5 file. '
                  'Will output all barcodes, and proceed as if '
                  'analyzed_barcodes_only=False')

    # Construct the anndata object.
    adata = anndata.AnnData(X=X,
                            obs={'barcode': d.pop('barcodes').astype(str)},
                            var={'gene_name': (d.pop('gene_names') if 'gene_names' in d.keys()
                                               else d.pop('name')).astype(str)},
                            dtype=X.dtype)
    adata.obs.set_index('barcode', inplace=True)
    adata.var.set_index('gene_name', inplace=True)

    # For CellRanger v2 legacy format, "gene_ids" was called "genes"... rename this
    if 'genes' in d.keys():
        d['id'] = d.pop('genes')

    # For purely aesthetic purposes, rename "id" to "gene_id"
    if 'id' in d.keys():
        d['gene_id'] = d.pop('id')

    # If genomes are empty, try to guess them based on gene_id
    if 'genome' in d.keys():
        if np.array([s.decode() == '' for s in d['genome']]).all():
            if '_' in d['gene_id'][0].decode():
                print('Genome field blank, so attempting to guess genomes based on gene_id prefixes')
                d['genome'] = np.array([s.decode().split('_')[0] for s in d['gene_id']], dtype=str)

    # Add other information to the anndata object in the appropriate slot.
    _fill_adata_slots_automatically(adata, d)

    # Add a special additional field to .var if it exists.
    if 'features_analyzed_inds' in adata.uns.keys():
        adata.var['cellbender_analyzed'] = [True if (i in adata.uns['features_analyzed_inds'])
                                            else False for i in range(adata.shape[1])]

    if analyzed_barcodes_only:
        for col in adata.obs.columns[adata.obs.columns.str.startswith('barcodes_analyzed')
                                     | adata.obs.columns.str.startswith('barcode_indices')]:
            try:
                del adata.obs[col]
            except Exception:
                pass
    else:
        # Add a special additional field to .obs if all barcodes are included.
        if 'barcodes_analyzed_inds' in adata.uns.keys():
            adata.obs['cellbender_analyzed'] = [True if (i in adata.uns['barcodes_analyzed_inds'])
                                                else False for i in range(adata.shape[0])]

    return adata


def dict_from_h5(file: str) -> Dict[str, np.ndarray]:
    """Read in everything from an h5 file and put into a dictionary."""
    d = {}
    with tables.open_file(file) as f:
        # read in everything
        for array in f.walk_nodes("/", "Array"):
            d[array.name] = array.read()
    return d


def _fill_adata_slots_automatically(adata, d):
    """Add other information to the adata object in the appropriate slot."""

    for key, value in d.items():
        try:
            if value is None:
                continue
            value = np.asarray(value)
            if len(value.shape) == 0:
                adata.uns[key] = value
            elif value.shape[0] == adata.shape[0]:
                if (len(value.shape) < 2) or (value.shape[1] < 2):
                    adata.obs[key] = value
                else:
                    adata.obsm[key] = value
            elif value.shape[0] == adata.shape[1]:
                if value.dtype.name.startswith('bytes'):
                    adata.var[key] = value.astype(str)
                else:
                    adata.var[key] = value
            else:
                adata.uns[key] = value
        except Exception:
            print('Unable to load data into AnnData: ', key, value, type(value))

def anndata_from_matrix(matrix_folder, write=False):
    raw_dict = defaultdict(lambda: "Not Present")
    for root, sample_list, filenames in os.walk(f'{matrix_folder}'):
        for sample_name in sample_list:
            sample = sample_name.split("\.")[0]
            raw_dict[sample] = sc.read_10x_mtx(f'{matrix_folder}/{sample_name}/')
    ad_all = anndata.concat(list(raw_dict.values()), label='sample', keys=list(raw_dict.keys()), join='outer', index_unique='-', merge='same')
    if write:
        ad_all.write_zarr(f"{write}/raw.zarr")
    return ad_all

def read_dask(store):
    f = zarr.open(store, mode="r")

    def callback(func, elem_name: str, elem, iospec):
        if iospec.encoding_type in (
            "dataframe",
            "csr_matrix",
            "csc_matrix",
            "awkward-array",
        ):
            # Preventing recursing inside of these types
            return read_elem(elem)
        elif iospec.encoding_type == "array":
            return da.from_zarr(elem)
        else:
            return func(elem)

    adata = read_dispatched(f, callback=callback)

    return adata

## doublet
def doublet_plot(basedir, sample_name, sample):
    doublet_score = sample.obs['doublet_score']
    sim_scores = sample.uns['scrublet']['doublet_scores_sim']
    plt.figure()
    n, bins, patches = plt.hist(sim_scores, 50, density=True, facecolor='g', alpha=0.75)
    n, bins, patches = plt.hist(doublet_score, 50, density=True, facecolor='b', alpha=0.75)
    plt.axvline(x=0.12)
    plt.savefig(os.path.join(basedir, f'figures/doublet_{sample_name}_score.pdf'), bbox_inches='tight')

## qc
def qc(data, name, basedir, flags={"mt": r"^MT-", "ribo": r"^RP[LS]", "hb": r"^HB"}, order=None, batch_key=None):
    """\
        Parameters
        -------
        data
            anndata object
        name
            name to identify your object
        batch_key
            batch key if there's any
    """
    qc_vars = []
    for flag, pattern in flags.items():
        if flag not in data.var.columns and pattern:
            pat = re.compile(pattern)
            data.var[flag] = np.array(
                [bool(pat.search(g)) for g in data.var_names]
            )
        if flag in data.var.columns:
            qc_vars.append(flag)

    sc.pp.calculate_qc_metrics(data, qc_vars=qc_vars, percent_top=None, log1p=False, inplace=True)

    axes = sc.pl.violin(data, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_ribo'], 
                 split=False, stripplot=False, use_raw=False, multi_panel=True, order=order,
                 groupby=batch_key, show=False)
    if batch_key:
        axes[1].set_ylim(0,50000)
        [ax.set_xticklabels(ax.get_xticklabels(), rotation = 45) for ax in axes]
    else:
        axes = axes.axes
        axes[0,0].set_ylim(0,10000)
        axes[0,1].set_ylim(0,50000)
        axes[0,2].set_ylim(0,20)
        axes[0,3].set_ylim(0,20)
    plt.savefig(os.path.join(basedir, f'figures/{name}_qc.pdf'),bbox_inches='tight')
    
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, constrained_layout=True)
    ax1_dict = sc.pl.scatter(data, x='total_counts', y='pct_counts_mt',ax=ax1, show=False, color=batch_key)
    ax2_dict = sc.pl.scatter(data, x='total_counts', y='n_genes_by_counts', ax=ax2, show=False, color=batch_key)
    ax3_dict = sc.pl.scatter(data, x='total_counts', y='pct_counts_ribo', ax=ax2, show=False, color=batch_key)
    plt.savefig(os.path.join(basedir, f'figures/{name}_mt_tc_ribo.pdf'),bbox_inches='tight')         
    
def filter_adata(adata, min_genes, percent, max_genes, percent_mt=1, filter_mt=True, min_counts=0):
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, max_genes=max_genes)
    adata_filter = adata[adata.obs.pct_counts_mt < percent_mt, :] 
    if percent>1:
        sc.pp.filter_genes(adata_filter, min_cells=percent)
    else:
        sc.pp.filter_genes(adata_filter, min_cells=percent*adata_filter.n_obs)
    if filter_mt:
        r = re.compile("^MT-")
        mt_adata = list(filter(r.match, adata_filter.var.index.values))
        adata_filter = adata_filter[:,~adata_filter.var.index.isin(mt_adata)]
    return adata_filter

def save_layer(adata, layer):
    adata.layers[layer] = adata.X.copy() 
    adata.X = adata.layer['raw'].copy()

def norm_hvg(adata, name, target_sum, n_top_genes=None):
    """\
        Parameters
        -------
        adata
            anndata object
    """ 
    ## Total-count normalize (library-size correct) the data matrix 𝐗 to 10,000 reads per cell
    sc.pp.normalize_total(adata, target_sum=target_sum, exclude_highly_expressed = True)
    ## Logarithmize the data:
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=n_top_genes)
    logging.info(f"Highly variable genes: {sum(adata.var.highly_variable)}")
    sc.pl.highly_variable_genes(adata, save=f'{name}_hvg.pdf')

def cell_cycle_analysis(cell_cycle_genes,adata,name):
    """\
        Parameters
        -------
        cell_cycle_genes
            predefined cell cyle datasets in this util
        adata
            anndata object
        name
            sample name
    """   
    s_genes = cell_cycle_genes[:43]
    g2m_genes = cell_cycle_genes[43:]
    cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
    scdata_cc_genes = adata[:, cell_cycle_genes]
    sc.tl.pca(scdata_cc_genes)
    sc.pl.pca_scatter(scdata_cc_genes, color='phase',save=f'{name}_cell_cycle.pdf')
## work on adata.X
def pca(adata, name, n_comps, pearson):
    if pearson:
        sc.experimental.pp.recipe_pearson_residuals(
        adata, n_top_genes=2000, batch_key='sample', n_comps=n_comps)
    else:
        sc.pp.pca(adata, n_comps=n_comps, use_highly_variable=True, svd_solver='arpack')
    pca = adata.obsm['X_pca']
    pca = pca/pca[:,0].std()
    ## is basedir necessary?
    np.save(f'data/{name}_pca{n_comps}.npy', pca)
    return adata

def tsne_and_umap(adata, name, n_comps, pearson, key=None):
    adata = pca(adata, name, n_comps, pearson)
    sc.pl.pca_overview(adata, save=f'{name}_clean_pca_overview.pdf')
    sc.pl.pca_variance_ratio(adata, save=f'{name}_npc.pdf')
    sc.pp.neighbors(adata, n_pcs =n_comps)
    sc.tl.umap(adata, min_dist=0.1, n_components=2, spread=0.5, maxiter=None, alpha=1.0)
    sc.tl.tsne(adata, n_pcs = n_comps)

    fig, axs = plt.subplots(1, 2, figsize=(8,4),constrained_layout=True)
    #sc.pl.tsne(corr_data, color="batchs", title="MNN tsne", ax=axs[0,0], show=False)
    sc.pl.tsne(adata, color=key, title="tsne", ax=axs[0], show=False)
    #sc.pl.umap(corr_data, color="batchs", title="MNN umap", ax=axs[1,0], show=False)
    sc.pl.umap(adata, color=key, title="umap", ax=axs[1], show=False)
    plt.savefig(f'figures/{name}_projection.pdf',bbox_inches='tight')
    
def find_markers(adata, methods, cluster, reference, celltype):
    for method in methods:
        sc.tl.rank_genes_groups(adata, cluster, method=method, key_added=method, use_raw=False, layer='log_norm', reference=reference)
        if method == 'logreg':
            df = pd.DataFrame(adata.uns[method]['names']).head(25)
        else:
            df = sc.get.rank_genes_groups_df(adata, group=None, pval_cutoff=0.05, log2fc_min =1, key = method)
        df.to_csv(f'output/DEGs/rank_genes/top_markers_{celltype}_{method}_{cluster}.tsv', sep='\t', quoting=3, header=False, index=False)
        sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key = method, save=f'{celltype}_{method}_{cluster}_genes.pdf')


def rgba_hex_colors_to_rgb(adata):
    for key in adata.uns.keys():
        if key.endswith('colors'):
            if type(adata.uns[key]) == dict:
                new_dict = defaultdict(list)
                for k,v in adata.uns[key].items():
                    new_dict[k] = np.array_str(np.array(
                [(v if len(v) <= 7 else v[:-2])]), precision = 6, suppress_small = True)
                adata.uns[key] = new_dict
            else:
                adata.uns[key] = np.array_str(np.array(
                [(c if len(c) <= 7 else c[:-2]) for c in adata.uns[key]]), precision = 6, suppress_small = True) 
    return adata

def write_adata(adata, file_name):
    ## adata.obs as str
    adata.obs = adata.obs.astype(str)
    
    adata = rgba_hex_colors_to_rgb(adata)
    ## pop out log1p
    adata.uns.pop('log1p', None)
    ## header too long?
    adata.write(file_name, compression='gzip')