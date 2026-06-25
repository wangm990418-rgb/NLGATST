"""Run NLGATST clustering on spatial transcriptomics data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_GPU_ID = "1"
DEFAULT_R_HOME = '/mnt/402f169c-136d-4c4c-b598-a9732ee96752/user/anaconda3/envs/NP1/lib/R'


def _configure_runtime_from_cli():
    """Set environment variables before importing torch or rpy2-dependent modules."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu_id", type=str, default=DEFAULT_GPU_ID)
    parser.add_argument("--r_home", type=str, default=DEFAULT_R_HOME)
    args, _ = parser.parse_known_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.r_home is not None:
        os.environ["R_HOME"] = args.r_home


_configure_runtime_from_cli()

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import stlearn as st
import torch
from matplotlib.lines import Line2D
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score

from NonlinearGAT.network_training import Training
from NonlinearGAT.utils import (
    PAGA_trajectory_inference_and_umap_visualization,
    fix_seed,
    mclust_R,
)
from utilities import parameter_setting

DEFAULT_PALETTE = [
    "#4A4F7E",
    "#BDAFC6",
    "#B5AF8B",
    "#D19246",
    "#71A682",
    "#81989B",
    "#4198AC",
]


def preprocessing(args):
    """Load spatial data and retain highly variable genes for model training."""
    args.inputPath = Path(args.inputPath or args.basePath)
    args.outPath = Path(f"{args.basePath}_NLGATST")
    args.outPath.mkdir(parents=True, exist_ok=True)

    if args.platform == "Visium":
        adata = sc.read_visium(args.inputPath, load_images=args.image == "T")
    else:
        adata = sc.read_h5ad(args.inputPath)

    adata.var_names_make_unique()
    _ensure_spatial_coordinates(adata)
    adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=np.float32)

    sc.pp.filter_genes(adata, min_cells=5)
    adata.layers["count"] = adata.X.copy()
    adata.raw = adata.copy()
    # sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)#ad
    sc.pp.normalize_total(adata, target_sum=1e4, exclude_highly_expressed=True, inplace=False)
    sc.pp.log1p(adata)
    _replace_invalid_values(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=args.n_top_genes)
    adata = adata[:, adata.var.highly_variable].copy()

    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.platform == "Visium":
        adata = st.convert_scanpy(adata)
    print(f"Preprocessed {adata.n_vars} genes and {adata.n_obs} spots.")
    return adata


def _ensure_spatial_coordinates(adata):
    if "spatial" in adata.obsm:
        return

    candidates = [
        ("imagerow", "imagecol"),
        ("pxl_row_in_fullres", "pxl_col_in_fullres"),
        ("pxl_row", "pxl_col"),
        ("row", "col"),
        ("x", "y"),
    ]
    for row_key, col_key in candidates:
        if row_key in adata.obs.columns and col_key in adata.obs.columns:
            adata.obsm["spatial"] = np.c_[
                adata.obs[row_key].to_numpy(dtype=float),
                adata.obs[col_key].to_numpy(dtype=float),
            ].astype("float32")
            return
    raise ValueError("No spatial coordinates were found in adata.obsm or adata.obs.")


def _replace_invalid_values(adata):
    if sparse.issparse(adata.X):
        bad = np.isnan(adata.X.data) | np.isinf(adata.X.data)
        if bad.any():
            adata.X.data[bad] = 0.0
    else:
        adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)


def add_ground_truth(adata, args):
    """Attach reference labels when available."""
    if args.label != "T":
        return

    if args.platform == "Visium":
        label_path = Path(args.ground_truth_dir) / f"cluster_labels_{args.dataset_name}.csv"
        ground_truth = pd.read_csv(label_path, index_col=0)
        adata.obs["ground_truth"] = list(ground_truth["ground_truth"])
    elif "label" in adata.obs:
        adata.obs["ground_truth"] = adata.obs["label"].copy()
    else:
        raise ValueError("Ground-truth labels were requested but no label source was found.")


def cluster_embedding(adata, args):
    """Cluster learned embeddings with mclust and store labels in ``pred_label``."""
    adata = mclust_R(adata, num_cluster=args.n_clusters, used_obsm="NLGATST_embed")
    adata.obs["pred_label"] = adata.obs["NLGATST_embed"].astype(str)
    return adata


def evaluate_clustering(adata, has_labels):
    """Compute internal metrics and supervised metrics when labels exist."""
    x_embed = adata.obsm["NLGATST_embed"]
    scores = {
        "silhouette": silhouette_score(x_embed, adata.obs["pred_label"]),
        "davies_bouldin": davies_bouldin_score(x_embed, adata.obs["pred_label"]),
    }
    if has_labels:
        obs = adata.obs.dropna(subset=["ground_truth", "pred_label"])
        scores["ari"] = adjusted_rand_score(obs["ground_truth"], obs["pred_label"])
        scores["nmi"] = normalized_mutual_info_score(obs["ground_truth"], obs["pred_label"])
    return scores


def export_class_expression(adata, out_dir):
    """Save class membership and mean gene expression by predicted class."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = adata.obs["pred_label"].astype(str).values
    classes = np.sort(pd.unique(labels)).astype(str)
    barcodes_by_class = {label: adata.obs_names[labels == label].tolist() for label in classes}
    idx_by_class = {label: np.where(labels == label)[0] for label in classes}

    pd.Series({label: len(items) for label, items in barcodes_by_class.items()}).to_csv(
        out_dir / "class_sizes.csv"
    )
    with open(out_dir / "barcodes_by_class.txt", "w", encoding="utf-8") as handle:
        for label in classes:
            handle.write(f"[{label}] n={len(barcodes_by_class[label])}\n")
            handle.write("\n".join(barcodes_by_class[label]) + "\n\n")

    if adata.raw is not None:
        x_all = adata.raw.X
        genes = adata.raw.var_names
    else:
        x_all = adata.layers["count"] if "count" in adata.layers else adata.X
        genes = adata.var_names

    class_means = []
    for label in classes:
        idx = idx_by_class[label]
        x_class = x_all[idx]
        if sparse.issparse(x_class):
            mean_expr = np.asarray(x_class.sum(axis=0)).ravel() / max(len(idx), 1)
        else:
            mean_expr = np.asarray(x_class.mean(axis=0)).ravel()
        class_means.append(mean_expr)

    mean_expr = pd.DataFrame(np.vstack(class_means), index=classes, columns=genes).T
    mean_expr.to_csv(out_dir / "gene_by_class_mean_expression.csv")

    eps = 1e-12
    specificity = np.vstack(class_means)
    specificity = specificity / (specificity.sum(axis=0, keepdims=True) + eps)
    pd.DataFrame(specificity, index=classes, columns=genes).T.to_csv(
        out_dir / "gene_by_class_specificity_proportion.csv"
    )


def map_clusters_to_ground_truth(adata, out_dir):
    """Save majority-vote and Hungarian mappings between predicted and reference labels."""
    table = pd.crosstab(adata.obs["ground_truth"].astype(str), adata.obs["pred_label"].astype(str))
    majority = pd.DataFrame(
        {
            "pred_cluster": table.columns,
            "mapped_region": table.idxmax(axis=0).values,
            "purity": (table.max(axis=0) / table.sum(axis=0)).values,
        }
    ).set_index("pred_cluster")

    cost = table.values.max() - table.values
    rows, cols = linear_sum_assignment(cost)
    hungarian = {table.columns[col]: table.index[row] for row, col in zip(rows, cols)}

    out_dir = Path(out_dir)
    table.to_csv(out_dir / "cluster_ground_truth_crosstab.csv")
    majority.to_csv(out_dir / "cluster_majority_mapping.csv")
    pd.Series(hungarian, name="mapped_region").to_csv(out_dir / "cluster_hungarian_mapping.csv")

    adata.obs["pred_region_majority"] = adata.obs["pred_label"].map(majority["mapped_region"])
    adata.obs["pred_region_hungarian"] = adata.obs["pred_label"].map(hungarian)
    print(
        "Mapping accuracy | majority: "
        f"{(adata.obs['pred_region_majority'] == adata.obs['ground_truth']).mean():.3f} | "
        f"hungarian: {(adata.obs['pred_region_hungarian'] == adata.obs['ground_truth']).mean():.3f}"
    )


def plot_spatial_labels(adata, label_key, out_path, title=None, marker="o"):
    """Plot categorical labels in spatial coordinates."""
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    x, y = coords[:, 0], -coords[:, 1]
    labels = adata.obs[label_key].astype("category")
    codes = labels.cat.codes.to_numpy()
    n_levels = len(labels.cat.categories)
    palette = [matplotlib.colors.to_rgba(DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)]) for i in range(n_levels)]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=8, marker=marker, edgecolors="none", c=[palette[i] for i in codes])
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if title:
        fig.suptitle(title, x=0.48, y=0.98, ha="center")

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=7, markerfacecolor=palette[i],
               markeredgecolor="none", label=label)
        for i, label in enumerate(labels.cat.categories.astype(str))
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    fig.savefig(f"{out_path}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_path}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = parameter_setting()
    args = parser.parse_args()
    fix_seed(args.seed)

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    adata = preprocessing(args)
    add_ground_truth(adata, args)

    Training(num=0, args=args, adata=adata)
    print(f"Embedding shape: {adata.obsm['NLGATST_embed'].shape}")

    adata = cluster_embedding(adata, args)
    scores = evaluate_clustering(adata, has_labels=args.label == "T")
    print("Metrics:", {key: round(value, 5) for key, value in scores.items()})

    if "ari" in scores and "nmi" in scores:
        title = f"NLGATST (ARI={scores['ari']:.3f}, NMI={scores['nmi']:.3f})"
    else:
        title = "NLGATST (ARI=NA, NMI=NA)"
    if args.label == "T":
        plot_spatial_labels(adata, "ground_truth", args.outPath / "ground_truth", title="Annotation")
    plot_spatial_labels(adata, "pred_label", args.outPath / "nlgatst_spatial_domains", title=title, marker="h")

    PAGA_trajectory_inference_and_umap_visualization(adata, args)

    adata.write_h5ad(args.outPath / "nlgatst_result.h5ad")
    print(f"Results saved to {args.outPath}")


if __name__ == "__main__":
    main()
