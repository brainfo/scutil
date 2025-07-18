"""
dotplot_utils.py
=================
Helpers for **Scanpy** dot‑plots where:

* **Colour** = row‑centred *z‑score* of mean log‑normalised expression.
* **Size**   = fraction of cells with expression > 0 in each group.

Key features
------------
* Works with Scanpy ≥ 1.8 and 1.10 regardless of where
  ``sc.get.aggregate`` stores statistics (``.X`` *or* ``.layers``).
* Accepts **``layer`` / ``use_raw`` / ``adata.X``** just like Scanpy plotting
  functions.
* Optional ``max_value`` clips extreme z‑scores (*à la* ``sc.pp.scale``).
* One‑liner flag to move gene labels to the right.
* Returns the **DotPlot** object so you can chain more styling.
"""
from __future__ import annotations

from typing import Sequence, Mapping, Any
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from scanpy.pl import DotPlot

__all__ = ["custom_deg_dotplot"]

# -----------------------------------------------------------------------------
# internal helpers
# -----------------------------------------------------------------------------

def _aggregate_expression(
    adata: sc.AnnData,
    genes: Sequence[str],
    groupby: str,
    *,
    layer: str | None = None,
    use_raw: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return mean expression & fraction‑expressed for *genes × groups*.

    The function is resilient to the Scanpy version:
    * Scanpy ≥ 1.10 → the **first** function result is in ``.X``; extras in
      ``.layers`` keyed by function name.
    * Scanpy ≤ 1.9  → every statistic lives in ``.layers``.
    """

    # --- gather both statistics in one call if the version supports it ------
    try:
        agg = sc.get.aggregate(
            adata[:, genes],
            by=groupby,
            layer=layer,
            use_raw=use_raw,
            func=["mean", "count_nonzero"],
        )
    except TypeError:  # Scanpy < 1.9 didn’t accept list for func
        agg_mean = sc.get.aggregate(
            adata[:, genes],
            by=groupby,
            layer=layer,
            use_raw=use_raw,
            func="mean",
        )
        agg_cnt = sc.get.aggregate(
            adata[:, genes],
            by=groupby,
            layer=layer,
            use_raw=use_raw,
            func="count_nonzero",
        )

        # unify index/columns order just in case
        agg_cnt = agg_cnt[agg_mean.obs_names, :][:, agg_mean.var_names]

        mean_arr = agg_mean.X if agg_mean.X is not None else agg_mean.layers["mean"]
        cnt_arr = (
            agg_cnt.layers["count_nonzero"]
            if "count_nonzero" in agg_cnt.layers
            else agg_cnt.X
        )
        n_obs = agg_mean.obs["n_obs"].to_numpy()[:, None]
        mean_df = pd.DataFrame(mean_arr, index=agg_mean.obs_names, columns=agg_mean.var_names)
        pct = cnt_arr / n_obs
        return mean_df, pct

    # ---------------- Scanpy ≥ 1.10 path ------------------------------------
    if agg.X is not None:
        mean_arr = agg.X
    elif "mean" in agg.layers:
        mean_arr = agg.layers["mean"]
    else:
        raise RuntimeError("Could not locate mean expression in aggregate result.")

    if "count_nonzero" in agg.layers:
        cnt_arr = agg.layers["count_nonzero"]
    else:
        warnings.warn(
            "count_nonzero not found in aggregate result; falling back to raw computation.",
            RuntimeWarning,
            stacklevel=2,
        )
        # compute manually (rare path; may be memory‑heavy for huge matrices)
        mat = (adata[:, genes].layers[layer] if layer else adata[:, genes].X) if not use_raw else adata[:, genes].raw.X  # type: ignore
        cnt_arr = (
            sc.get.aggregate(
                adata[:, genes].copy(),
                by=groupby,
                func="count_nonzero",
                layer=layer,
                use_raw=use_raw,
            ).X
        )

    n_obs = agg.obs["n_obs"].to_numpy()[:, None]
    mean_df = pd.DataFrame(mean_arr, index=agg.obs_names, columns=agg.var_names)
    pct = cnt_arr / n_obs
    return mean_df, pct


def _zscore(df: pd.DataFrame, max_value: float | None = None) -> pd.DataFrame:
    """Return *row* centred and variance‑scaled copy, with optional clipping."""
    z = (df - df.mean(0)) / df.std(0).replace(0, np.nan)
    if max_value is not None:
        z = z.clip(-max_value, max_value)
    return z

# -----------------------------------------------------------------------------
# public API
# -----------------------------------------------------------------------------

def custom_deg_dotplot(
    adata: sc.AnnData,
    genes: Sequence[str],
    groupby: str,
    *,
    layer: str | None = "log_norm",
    use_raw: bool = False,
    max_value: float | None = None,
    right_labels: bool = False,
    cmap: str | Any = "RdBu_r",
    size_title: str | None = "Fraction of cells (%)",
    colorbar_title: str | None = "Row Z-score (log-norm)",
    dotplot_kwargs: Mapping[str, Any] | None = None,
) -> DotPlot:
    """Create a DEG dot‑plot: Z‑score colours, %‑expressed sizes.

    Parameters
    ----------
    adata, genes, groupby
        Standard Scanpy semantics.
    layer, use_raw
        Expression source selection.
    max_value
        Clip |z| to this value (helps avoid a few extreme dots dominating the
        colour bar). ``None`` → no clipping.
    right_labels
        ``True`` → move gene names (y‑ticks) to the right of the plot.
    cmap
        A Matplotlib colormap *object* or *name*. E.g. build one via
        ``LinearSegmentedColormap.from_list`` and pass it.
    size_title, colorbar_title
        Legends; set to ``None`` to keep Scanpy defaults.
    dotplot_kwargs
        Forwarded to :class:`scanpy.pl.DotPlot`, e.g. ``dendrogram=True``.
    """
    dotplot_kwargs = {} if dotplot_kwargs is None else dict(dotplot_kwargs)

    # 1  aggregate stats
    mean_df, pct = _aggregate_expression(
        adata, genes, groupby, layer=layer, use_raw=use_raw
    )

    # 2  z‑scores
    z_df = _zscore(mean_df, max_value=max_value)

    # 3  build DotPlot
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

    # 4  y‑tick repositioning
    if right_labels:
        ax = dp.get_axes()["mainplot_ax"]
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", labelright=True, labelleft=False, pad=2)
        ax.figure.subplots_adjust(right=0.82)

    return dp

# -----------------------------------------------------------------------------
# CLI demo (run `python dotplot_utils.py my_data.h5ad` to test)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from matplotlib.colors import LinearSegmentedColormap

    parser = argparse.ArgumentParser(description="Demo custom DEG dot‑plot.")
    parser.add_argument("adata", help="Annotated .h5ad file with log_norm layer")
    parser.add_argument("--groupby", default="group3", help="obs column")
    parser.add_argument("--genes", nargs="*", help="genes to plot; default top‑20")
    parser.add_argument("--raw", action="store_true", help="use adata.raw")
    parser.add_argument("--out", default="deg_dotplot.pdf", help="output file")
    args = parser.parse_args()

    ad = sc.read_h5ad(args.adata)
    gene_list = args.genes or list(ad.uns["rank_genes_groups"]["names"][0][:20])

    # custom teal‑white‑red palette if you like
    cmap = LinearSegmentedColormap.from_list("twr", ["#468189", "white", "#FF0022"])

    dp = custom_deg_dotplot(
        ad,
        genes=gene_list,
        groupby=args.groupby,
        use_raw=args.raw,
        layer=None if args.raw else "log_norm",
        max_value=3,
        right_labels=True,
        cmap=cmap,
    )
    dp.show()
    dp.savefig(args.out)
