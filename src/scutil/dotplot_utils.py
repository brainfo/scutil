"""
dotplot_utils.py
=================
Helpers for **Scanpy** DEG dot‑plots where:

* **Colour** = row‑centred *z‑scores* (mean−centered, σ‑scaled).
* **Size**   = fraction of cells with expression > 0 per group.

Key points
----------
* Works with Scanpy 1.8 → 1.10 (handles both aggregate APIs).
* Optional ``max_value`` clips extreme z‑scores.
* ``right_labels=True`` moves gene labels to the right.
* Returns the **DotPlot** object for further styling or saving.
"""
from __future__ import annotations

from typing import Sequence, Mapping, Any
import numpy as np
import pandas as pd
import scanpy as sc
from scanpy.pl import DotPlot

__all__ = ["custom_deg_dotplot"]

# -----------------------------------------------------------------------------
# internal helpers
# -----------------------------------------------------------------------------

def _group_sizes(adata: sc.AnnData, groupby: str, order: Sequence[str]) -> np.ndarray:
    """Return cell counts per group in *order* (shape = G×1)."""
    return (
        adata.obs[groupby]
        .value_counts()
        .reindex(order)
        .fillna(0)
        .to_numpy(int)[:, None]
    )


def _aggregate_expression(
    adata: sc.AnnData,
    genes: Sequence[str],
    groupby: str,
    *,
    layer: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute mean expression & fraction‑expressed (0–1) for each group/ gene."""

    # try one call with both stats (Scanpy ≥1.9)
    try:
        agg = sc.get.aggregate(
            adata[:, genes],
            by=groupby,
            layer=layer,
            func=["mean", "count_nonzero"],
        )
        multi = True
    except TypeError:  # older Scanpy
        multi = False

    if multi:
        mean_arr = agg.X if agg.X is not None else agg.layers["mean"]
        cnt_arr = agg.layers["count_nonzero"]
        n_obs = (
            agg.obs.get("n_obs").to_numpy()[:, None]
            if "n_obs" in agg.obs
            else _group_sizes(adata, groupby, agg.obs_names)
        )
        mean_df = pd.DataFrame(mean_arr, index=agg.obs_names, columns=agg.var_names)
        pct = cnt_arr / n_obs
        return mean_df, pct

    # fallback: two calls for mean & count_nonzero
    agg_mean = sc.get.aggregate(
        adata[:, genes],
        by=groupby,
        layer=layer,
        func="mean",
    )
    agg_cnt = sc.get.aggregate(
        adata[:, genes],
        by=groupby,
        layer=layer,
        func="count_nonzero",
    )

    # align order just in case
    agg_cnt = agg_cnt[agg_mean.obs_names, :][:, agg_mean.var_names]

    mean_arr = agg_mean.X if agg_mean.X is not None else agg_mean.layers["mean"]
    cnt_arr = (
        agg_cnt.layers["count_nonzero"]
        if "count_nonzero" in agg_cnt.layers
        else agg_cnt.X
    )
    n_obs = (
        agg_mean.obs.get("n_obs").to_numpy()[:, None]
        if "n_obs" in agg_mean.obs.columns
        else _group_sizes(adata, groupby, agg_mean.obs_names)
    )
    mean_df = pd.DataFrame(mean_arr, index=agg_mean.obs_names, columns=agg_mean.var_names)
    pct = cnt_arr / n_obs
    return mean_df, pct


def _zscore(df: pd.DataFrame, max_value: float | None) -> pd.DataFrame:
    z = (df - df.mean(0)) / df.std(0).replace(0, np.nan)
    return z.clip(-max_value, max_value) if max_value is not None else z

# -----------------------------------------------------------------------------
# public API
# -----------------------------------------------------------------------------

def custom_deg_dotplot(
    adata: sc.AnnData,
    genes: Sequence[str],
    groupby: str,
    *,
    layer: str | None = "log_norm",
    max_value: float | None = None,
    right_labels: bool = False,
    cmap: str | Any = "RdBu_r",
    size_title: str | None = "Fraction of cells (%)",
    colorbar_title: str | None = "Row Z‑score (log‑norm)",
    dotplot_kwargs: Mapping[str, Any] | None = None,
) -> DotPlot:
    """Return a Scanpy ``DotPlot`` whose colours = z‑scores & sizes = % cells."""
    dotplot_kwargs = dict(dotplot_kwargs or {})

    mean_df, pct = _aggregate_expression(adata, genes, groupby, layer=layer)
    z_df = _zscore(mean_df, max_value)

    dp = DotPlot(
        adata,
        var_names=genes,
        groupby=groupby,
        dot_color_df=z_df,
        dot_size_df=pct,
        **dotplot_kwargs,
    )
    dp.style(cmap=cmap, edgecolor="face")
    dp.legend(colorbar_title=colorbar_title, size_title=size_title)

    if right_labels:
        ax = dp.get_axes()["mainplot_ax"]
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", labelright=True, labelleft=False, pad=2)
        ax.figure.subplots_adjust(right=0.82)

    return dp

# -----------------------------------------------------------------------------
# CLI demo
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from matplotlib.colors import LinearSegmentedColormap

    parser = argparse.ArgumentParser(description="Demo custom DEG dot‑plot.")
    parser.add_argument("adata", help=".h5ad file with `log_norm` layer")
    parser.add_argument("--groupby", default="group3", help="obs column for groups")
    parser.add_argument("--genes", nargs="*", help="genes to plot (default: top‑20)")
    parser.add_argument("--layer", default="log_norm", help="layer or 'raw' or None")
    parser.add_argument("--out", default="deg_dotplot.pdf", help="output figure path")
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
    dp.show()
    dp.savefig(args.out)