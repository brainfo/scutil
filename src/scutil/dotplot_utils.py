"""
dotplot_utils.py
----------------
Utility helpers for Scanpy **dot‑plots** that

* colour dots with **row‑centred z‑scores** (mean‑subtracted, σ‑scaled),
* size dots by **fraction of cells expressed** (values > 0),
* optionally clip extreme z‑scores via ``max_value``,
* optionally move gene labels (y‑axis) to the right‑hand side,
* let you pick *any* expression source: ``use_raw=True`` (``adata.raw``), a
  named ``layer``, or ``adata.X``.

The main entry point is :pyfunc:`custom_deg_dotplot`, which returns the
:class:`scanpy.pl.DotPlot` object so you can style it further or save it.
"""
from __future__ import annotations

from typing import Sequence, Mapping, Any

import numpy as np
import pandas as pd
import scanpy as sc
from scanpy.pl import DotPlot

__all__ = [
    "custom_deg_dotplot",
]

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
    """Aggregate per‑group *mean expression* and *fraction expressed*.

    The implementation follows Scanpy ≥ 1.10 conventions:
    * ``sc.get.aggregate`` always puts the **first** function’s result in
      ``.X``.
    * Additional statistics are stored in ``.layers`` with the function
      name as key.

    Returns
    -------
    mean_df
        DataFrame (groups × genes) with mean expression.
    pct_matrix
        ``np.ndarray`` with fractions in the range 0‒1.
    """
    agg = sc.get.aggregate(
        adata[:, genes],
        by=groupby,
        layer=layer,
        use_raw=use_raw,
        func=["mean", "count_nonzero"],
    )

    # --- Scanpy stores the first statistic in .X ---------------------------
    mean_arr = agg.X.copy()

    # count_nonzero is in a layer when requested together with mean
    if "count_nonzero" in agg.layers:
        count_nz = agg.layers["count_nonzero"].copy()
    else:  # fallback: sc<1.10 or single‑stat call
        count_nz = agg.X.copy()

    n_obs = agg.obs["n_obs"].to_numpy()[:, None]  # (groups, 1)

    mean_df = pd.DataFrame(mean_arr, index=agg.obs_names, columns=agg.var_names)
    pct_matrix = count_nz / n_obs  # broadcast div → fraction 0‒1
    return mean_df, pct_matrix


def _zscore(df: pd.DataFrame, max_value: float | None = None) -> pd.DataFrame:
    """Row‑centre + variance‑scale *in a copy*; clip to ``±max_value`` if set."""
    z = (df - df.mean(0)) / df.std(0).replace(0, np.nan)
    if max_value is not None:
        z = z.clip(lower=-max_value, upper=max_value)
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
    size_title: str = "Fraction of cells (%)",
    colorbar_title: str = "Row Z-score (log-norm)",
    dotplot_kwargs: Mapping[str, Any] | None = None,
) -> DotPlot:
    """Create a DotPlot whose *colour* shows z‑scores and *size* shows % expressed.

    Parameters
    ----------
    adata
        Annotated data matrix.
    genes
        Ordered list of gene identifiers (must be in ``adata.var_names``).
    groupby
        Column in ``adata.obs`` used to group cells.
    layer, use_raw
        Expression source:
        * ``use_raw=True`` → ``adata.raw``; ``layer`` ignored.
        * ``layer=None``   → ``adata.X``.
        * ``layer='name'`` → ``adata.layers['name']``.
    max_value
        Clip absolute z‑score to this value (mimics ``sc.pp.scale(max_value)``).
    right_labels
        If *True*, show gene names on the right.
    cmap
        Any Matplotlib colormap or a name understood by Matplotlib.
    dotplot_kwargs
        Passed verbatim to :class:`scanpy.pl.DotPlot` (e.g. ``dendrogram=True``).

    Returns
    -------
    DotPlot
        The configured DotPlot object; call ``.show()`` or ``.savefig()``.
    """
    dotplot_kwargs = {} if dotplot_kwargs is None else dict(dotplot_kwargs)

    # 1  aggregate statistics
    mean_df, pct = _aggregate_expression(
        adata, genes, groupby, layer=layer, use_raw=use_raw
    )

    # 2  compute row‑centred z‑scores
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

    # legends & style
    dp.legend(colorbar_title=colorbar_title, size_title=size_title)
    dp.style(cmap=cmap, edgecolor="face")

    # optional y‑label move
    if right_labels:
        main_ax = dp.get_axes()["mainplot_ax"]
        main_ax.yaxis.tick_right()
        main_ax.tick_params(axis="y", labelright=True, labelleft=False, pad=2)
        main_ax.figure.subplots_adjust(right=0.82)

    return dp


# -----------------------------------------------------------------------------
# CLI demo when the module is executed directly
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demo custom DEG dot‑plot.")
    parser.add_argument("adata_file", help="input .h5ad with log_norm layer & groups")
    parser.add_argument("--groupby", default="group3", help="obs column with groups")
    parser.add_argument("--genes", nargs="*", help="genes to plot; default = top‑20")
    parser.add_argument("--raw", action="store_true", help="use adata.raw as matrix")
    parser.add_argument("--out", default="deg_dotplot.pdf", help="output figure file")
    args = parser.parse_args()

    ad = sc.read_h5ad(args.adata_file)

    # default gene list = first 20 names from rank_genes_groups
    genes = args.genes or list(ad.uns["rank_genes_groups"]["names"][0][:20])

    dp_obj = custom_deg_dotplot(
        ad,
        genes=genes,
        groupby=args.groupby,
        layer=None if args.raw else "log_norm",
        use_raw=args.raw,
        max_value=3,
        right_labels=True,
    )
    dp_obj.show()
    dp_obj.savefig(args.out)