"""
dotplot_utils.py
----------------
Utility functions to build Scanpy `DotPlot`s that
• colour‑code mean expression as *row‑centred z‑scores*
• size dots by *fraction of cells expressed* (0‒1)
• optionally clip extreme z‑scores (``max_value``)
• optionally move gene labels to the right‑hand side of the plot.

The helpers return the **DotPlot** object so you can style, save or further
customise exactly as you would with any Scanpy figure.
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


def _aggregate_expression(
    adata: sc.AnnData,
    genes: Sequence[str],
    groupby: str,
    layer: str | None = "log_norm",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Aggregate **mean** and **count‑non‑zero** per group & gene.

    Parameters
    ----------
    adata
        AnnData with expression values.
    genes
        Ordered list of gene identifiers (index into *adata.var_names*).
    groupby
        Column in *adata.obs* that defines the groups.
    layer
        Where to read expression from. ``None`` means ``adata.X``;
        use ``"raw"`` to read from *adata.raw*.

    Returns
    -------
    mean_df
        DataFrame (groups × genes) of mean expression.
    pct_matrix
        Numpy array of shape (groups, genes) with the fraction of cells
        expressed (values in 0‒1).
    """
    agg = sc.get.aggregate(
        adata[:, genes],
        by=groupby,
        layer=layer,
        func=["mean", "count_nonzero"],
    )
    mean_df = pd.DataFrame(
        agg["mean"].X, index=agg.obs.index.copy(), columns=agg.var_names.copy()
    )
    pct_matrix = agg["count_nonzero"].X / agg["n_obs"].X  # 0‒1
    return mean_df, pct_matrix


def _zscore(df: pd.DataFrame, max_value: float | None = None) -> pd.DataFrame:
    """Row‑centre and variance‑scale *in place*; optionally clip to ``±max_value``."""
    z = (df - df.mean(0)) / df.std(0).replace(0, np.nan)
    if max_value is not None:
        z = z.clip(lower=-max_value, upper=max_value)
    return z


def custom_deg_dotplot(
    adata: sc.AnnData,
    genes: Sequence[str],
    groupby: str,
    *,
    layer: str | None = "log_norm",
    max_value: float | None = None,
    right_labels: bool = False,
    cmap: str = "RdBu_r",
    size_title: str = "Fraction of cells (%)",
    colorbar_title: str = "Row Z-score (log-norm)",
    dotplot_kwargs: Mapping[str, Any] | None = None,
) -> DotPlot:
    """Build and return a customised DotPlot.

    Parameters
    ----------
    adata
        Annotated data matrix.
    genes
        Ordered list of gene names to show on the *y*‑axis.
    groupby
        Column in *adata.obs* that defines the groups (one column per group).
    layer
        Expression layer to use (``None`` → ``adata.X``).
    max_value
        Clip absolute z‑score to this value (like ``sc.pp.scale(max_value=…)``).
    right_labels
        If *True*, move gene labels (tick labels) to the right‑hand side.
    cmap
        Matplotlib colormap name passed to ``DotPlot.style()``.
    size_title, colorbar_title
        Legend titles for dot size and colour, respectively.
    dotplot_kwargs
        Extra kwargs forwarded verbatim to ``scanpy.pl.DotPlot``.

    Returns
    -------
    DotPlot
        The configured Scanpy ``DotPlot`` object. Call ``.show()`` or further
        customise before saving.
    """
    dotplot_kwargs = {} if dotplot_kwargs is None else dict(dotplot_kwargs)

    # ------------------------------------------------------------------
    # 1. Aggregate per group
    mean_df, pct_matrix = _aggregate_expression(adata, genes, groupby, layer)

    # 2. Z‑score (centre + scale) and clip if requested
    z_df = _zscore(mean_df, max_value=max_value)

    # ------------------------------------------------------------------
    # 3. Build DotPlot object
    dp = DotPlot(
        adata,
        var_names=genes,
        groupby=groupby,
        dot_color_df=z_df,
        dot_size_df=pct_matrix,
        **dotplot_kwargs,
    )

    # 4. Legend styling
    dp.legend(colorbar_title=colorbar_title, size_title=size_title)

    # 5. Aesthetics — colormap & edge masking
    dp.style(cmap=cmap, edgecolor="face")

    # 6. Optional label repositioning
    if right_labels:
        main_ax = dp.get_axes()["mainplot_ax"]
        main_ax.yaxis.tick_right()
        main_ax.tick_params(axis="y", labelright=True, labelleft=False, pad=2)
        # Make room for the labels so they’re not clipped
        main_ax.figure.subplots_adjust(right=0.82)

    return dp


# -----------------------------------------------------------------------------
# Example usage when run as a script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demo custom DEG dot‑plot.")
    parser.add_argument("adata_file", help=".h5ad file with log_norm layer")
    parser.add_argument("--groupby", default="group3", help="obs column")
    parser.add_argument(
        "--genes",
        nargs="*",
        help="Genes to plot (default: top 20 from rank_genes_groups).",
    )
    parser.add_argument("--out", default="deg_dotplot.pdf", help="Output file")
    args = parser.parse_args()

    ad = sc.read_h5ad(args.adata_file)

    # fall back on top‑ranked genes if none provided
    genes = (
        args.genes
        if args.genes
        else list(ad.uns["rank_genes_groups"]["names"][0][:20])
    )

    dp_obj = custom_deg_dotplot(
        ad,
        genes=genes,
        groupby=args.groupby,
        layer="log_norm",
        max_value=3,
        right_labels=True,
    )
    dp_obj.show()
    dp_obj.savefig(args.out)
