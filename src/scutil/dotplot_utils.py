"""
dotplot_utils.py

"""
from __future__ import annotations

from typing import Mapping, Sequence, Any

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

__all__ = ["custom_deg_dotplot", "swap_axes"]

###############################################################################
# Helper functions
###############################################################################

def _group_sizes(adata: sc.AnnData, groupby: str, order: Sequence[str]) -> np.ndarray:
    """Cell counts per group in *order* (shape = G×1)."""
    return adata.obs[groupby].value_counts().reindex(order).fillna(0).to_numpy(int)[:, None]


def _aggregate_expression(
    adata: sc.AnnData,
    genes: Sequence[str],
    groupby: str,
    *,
    layer: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean expression and fraction‑expressed for each group × gene."""
    # Mean expression
    agg_mean = sc.get.aggregate(adata[:, genes], by=groupby, layer=layer, func="mean")
    mean_arr = agg_mean.X if agg_mean.X is not None else agg_mean.layers["mean"]
    mean_df = pd.DataFrame(mean_arr, index=agg_mean.obs_names, columns=agg_mean.var_names)

    # Non‑zero counts  
    agg_cnt = sc.get.aggregate(adata[:, genes], by=groupby, layer=layer, func="count_nonzero")
    agg_cnt = agg_cnt[mean_df.index, :][:, mean_df.columns]
    cnt_arr = agg_cnt.X if agg_cnt.X is not None else agg_cnt.layers["count_nonzero"]

    # Group sizes
    n_obs = (agg_cnt.obs["n_obs"].to_numpy(int)[:, None] 
             if "n_obs" in agg_cnt.obs.columns 
             else _group_sizes(adata, groupby, mean_df.index))

    pct_df = pd.DataFrame(cnt_arr / n_obs, index=mean_df.index, columns=mean_df.columns)
    return mean_df, pct_df


def _zscore(df: pd.DataFrame, max_value: float | None) -> pd.DataFrame:
    z = (df - df.mean(0)) / df.std(0).replace(0, np.nan)
    return z.clip(-max_value, max_value) if max_value else z

###############################################################################
# Public API
###############################################################################

def custom_deg_dotplot(
    adata: sc.AnnData,
    genes: Sequence[str],
    groupby: str,
    layer: str | None = "log_norm",
    max_value: float | None = None,
    # right_labels: bool = False,
    # cmap: str | Any = "RdBu_r",
    # size_title: str = "Fraction of cells (%)",
    # colorbar_title: str = "Z‑score (log‑norm)",
    swap_axes: bool = False,
    # dotplot_kwargs: Mapping[str, Any] | None = None,
    figsize: tuple
) -> DotPlot:
    """Return a Scanpy ``DotPlot`` with z‑score colours & %‑size dots."""
    mean_df, pct_df = _aggregate_expression(adata, genes, groupby, layer=layer)
    z_df = _zscore(mean_df, max_value)
    z_names = "genes", "group"
    pct_names = "proportion", "group"
    if swap_axes:
        z_df, pct_df = z_df.T, pct_df.T
        z_names = z_names[::-1]
        pct_names = pct_names[::-1]

    # z_l, pct_l = pd.melt(z_df,value_vars=z_df.columns, var_name=z_names[0], value_name=z_names[1]), pd.melt(pct_df, value_vars=pct_df.columns, var_name=pct_names[0], value_name=pct_names[1])
    # f, ax = plt.subplots(figsize=figsize)
    # plt.scatter(z_l[z_names[0]], z_l[z_names[1]], s=pct_names[1], c=)
    return z_df, pct_df


def swap_axes(dotplot: DotPlot) -> DotPlot:
    """Swap x and y axes of a DotPlot following Scanpy's approach."""
    # Toggle the swapped state like Scanpy does
    dotplot.are_axes_swapped = not getattr(dotplot, 'are_axes_swapped', False)
    return dotplot