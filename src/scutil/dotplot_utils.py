"""
dotplot_utils.py

"""
from __future__ import annotations

from typing import Mapping, Sequence, Any

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

__all__ = ["custom_deg_dotplot"]

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
    figsize: tuple,
    save: str,
    layer: str | None = "log_norm",
    max_value: float | None = None,
    cmap: str | Any = "RdBu_r",
    swap_axes: bool = False,
    y_right: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.collections.PathCollection:
    # --- data prep (unchanged) -------------------------------------------------
    mean_df, pct_df = _aggregate_expression(adata, genes, groupby, layer=layer)
    z_df          = _zscore(mean_df, max_value)

    names = ("genes", "group")
    if swap_axes:
        z_df, pct_df = z_df.T, pct_df.T
        names = names[::-1]

    z_df  = z_df.reset_index(names=names[1])
    pct_df = pct_df.reset_index(names=names[1])

    z_l   = pd.melt(z_df,  id_vars=names[1], value_name="z-score",
                    value_vars=[c for c in z_df.columns if c != names[1]],
                    var_name=names[0])
    pct_l = pd.melt(pct_df, id_vars=names[1], value_name="fraction",
                    value_vars=[c for c in pct_df.columns if c != names[1]],
                    var_name=names[0])

    # --- figure & main axis ----------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.margins(x=0.20)
    ax.margins(y=0.20)
    if y_right:
        ax.spines[['top', 'left']].set_visible(False)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        legend_bbox = (-0.05, 0.02)   # place legend just outside the axis
    else:
        ax.spines[['top', 'right']].set_visible(False)
        legend_bbox = (1.05, 0.02)

    # dots ----------------------------------------------------------------------
    scatter = ax.scatter(
        z_l[names[0]],
        z_l[names[1]],
        s=pct_l["fraction"] * 100,
        c=z_l["z-score"],
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        linewidths=0.5,
    )

    # --- fixed‑size colour‑bar -------------------------------------------------
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("bottom", size=0.1, pad=0.5)
    cbar    = fig.colorbar(scatter, cax=cax, orientation="horizontal")
    cbar.outline.set_visible(False)
    cbar.ax.set_title("z‑score", pad=5)

    # --- fixed‑size legend -----------------------------------------------------
    frac_sizes = [20, 60, 100]
    legend_handles = [
        plt.Line2D([], [], marker='o', linestyle='',
                   color='gray', alpha=0.6, markersize=(s**0.5))
        for s in frac_sizes
    ]
    legend_labels = [f"{s}%" for s in frac_sizes]

    # draw the legend *at the figure level* so it is excluded from tight_layout
    legend = fig.legend(
        legend_handles, legend_labels,
        title="Fraction",
        loc='center left',
        bbox_to_anchor=legend_bbox,
        frameon=False,
    )
    legend.set_in_layout(False)        # keep legend size *constant*

    fig.savefig(save, bbox_inches="tight", bbox_extra_artists=[legend], pad_inches=0)
    return scatter
