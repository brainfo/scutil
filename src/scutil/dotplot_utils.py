"""
dotplot_utils.py
----------------
Utility to build a **Scanpy DotPlot** where

* **colour** = row‑centred *z‑score* of mean log‑norm expression;
* **size**   = fraction of cells expressed (> 0) per group.

Key points
~~~~~~~~~~
* Works with Scanpy 1.8 → 1.10 (uses only stable API).
* Reads expression from `adata.X` (`layer=None`) **or** any named layer
  (`layer="log_norm"`, `layer="scaled"`, …).
* Optional `max_value` clips extreme z‑scores (like `sc.pp.scale`).
* `right_labels=True` moves gene labels (y‑ticks) to the right.
* Returns the :class:`scanpy.pl.DotPlot` object for further styling or saving.
"""
from __future__ import annotations

from typing import Mapping, Sequence, Any

import numpy as np
import pandas as pd
import scanpy as sc
from scanpy.pl import DotPlot

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
    *,
    layer: str | None = "log_norm",
    max_value: float | None = None,
    right_labels: bool = False,
    cmap: str | Any = "RdBu_r",
    size_title: str = "Fraction of cells (%)",
    colorbar_title: str = "Row Z‑score (log‑norm)",
    swap_axes: bool = False,
    dotplot_kwargs: Mapping[str, Any] | None = None,
) -> DotPlot:
    """Return a Scanpy ``DotPlot`` with z‑score colours & %‑size dots."""
    mean_df, pct_df = _aggregate_expression(adata, genes, groupby, layer=layer)
    z_df = _zscore(mean_df, max_value)

    dp = DotPlot(adata, var_names=genes, groupby=groupby, 
                 dot_color_df=z_df, dot_size_df=pct_df, **(dotplot_kwargs or {}))
    dp.style(cmap=cmap)
    dp.legend(colorbar_title=colorbar_title, size_title=size_title)

    if right_labels:
        ax = dp.get_axes()["mainplot_ax"]
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", labelright=True, labelleft=False, pad=2)
        ax.figure.subplots_adjust(right=0.82)

    if swap_axes:
        color_df = dp.dot_color_df.T if dp.dot_color_df is not None else None
        size_df = dp.dot_size_df.T if dp.dot_size_df is not None else None
        dp = DotPlot(adata, var_names=dp.groupby_names, groupby=dp.var_names,
                     dot_color_df=color_df, dot_size_df=size_df, **(dotplot_kwargs or {}))
        dp.style(cmap=cmap)
        dp.legend(colorbar_title=colorbar_title, size_title=size_title)

    return dp


def swap_axes(dotplot: DotPlot) -> DotPlot:
    """Swap x and y axes of a DotPlot by transposing the underlying data."""
    return DotPlot(
        dotplot.adata, var_names=dotplot.groupby_names, groupby=dotplot.var_names,
        dot_color_df=dotplot.dot_color_df.T, dot_size_df=dotplot.dot_size_df.T, **dotplot.kwds
    )

###############################################################################
# CLI demo (optional)
###############################################################################
if __name__ == "__main__":  # pragma: no cover
    import argparse
    from matplotlib.colors import LinearSegmentedColormap

    parser = argparse.ArgumentParser(description="Demo custom DEG dot‑plot.")
    parser.add_argument("adata", help=".h5ad file with log‑norm layer + groups")
    parser.add_argument("--groupby", default="group3", help="obs column")
    parser.add_argument("--genes", nargs="*", help="genes to plot (default top‑20)")
    parser.add_argument("--layer", default="log_norm", help="layer name or None")
    parser.add_argument("--out", default="deg_dotplot.pdf", help="output PDF path")
    args = parser.parse_args()

    ad = sc.read_h5ad(args.adata)
    genes = args.genes or list(ad.uns["rank_genes_groups"]["names"][0][:20])

    cmap = LinearSegmentedColormap.from_list("twr", ["#468189", "white", "#FF0022"])

    dp = custom_deg_dotplot(
        ad,
        genes=genes,
        groupby=args.groupby,
        layer=None if args.layer.lower() == "none" else args.layer,
        max_value=3,
        right_labels=True,
        cmap=cmap,
    )
    dp.savefig(args.out)
