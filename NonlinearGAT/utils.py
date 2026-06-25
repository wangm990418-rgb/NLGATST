"""Data preparation, graph construction, clustering, and visualization utilities."""

from __future__ import annotations

import os
import random
from itertools import cycle, islice

import anndata
import dgl
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from dgl import backend as dF
from scipy.sparse import csc_matrix, issparse
from scipy.spatial import distance_matrix
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.neighbors import kneighbors_graph
from torch.backends import cudnn

import gudhi


def load_data(adata):
    """Build the DGL graph and normalized feature matrix from an AnnData object."""
    print("Loading dataset from AnnData object...")
    adj_matrix = graph_alpha(adata.obsm["spatial"], n_neighbors=10)

    features = csc_matrix(adata.X.toarray() if issparse(adata.X) else adata.X)
    adj_1 = torch.FloatTensor(adj_matrix.toarray())

    adj = csc_matrix(adj_matrix)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    features = normalize_features(features)

    rows, cols = adj.nonzero()
    graph = dgl.graph((np.asarray(rows), np.asarray(cols)))
    graph.ndata["feat"] = dF.tensor(features.todense(), dtype=dF.data_type_dict["float32"])
    edge = torch.FloatTensor(np.array(adj.todense())).nonzero().t()
    return graph, edge, features, adj_1


def normalize_adj(mx):
    """Symmetrically normalize a sparse adjacency matrix."""
    row_sum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(row_sum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize a sparse feature matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum + 1e-9, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    return sp.diags(r_inv).dot(mx)


def calculate_ari(predicted_labels, true_labels):
    """Compute the adjusted Rand index."""
    return ari(true_labels, predicted_labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a SciPy sparse matrix to a Torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def training_performance(val_loss, best_val, epoch, best_epoch):
    """Track the lowest training loss observed so far."""
    if val_loss < best_val:
        return val_loss, epoch
    return best_val, best_epoch


def PAGA_trajectory_inference_and_umap_visualization(adata, args):
    """Generate UMAP and PAGA plots from the learned embedding."""
    palette_hex = [
        "#4E659B", "#8A8CBF", "#B8A8CF", "#E7BCC6", "#FDCF9E", "#EFA484", "#B6766C",
        "#6EA6D8", "#9DC4E7", "#C9DEF2", "#A0D6D1", "#71B7B2", "#49A69C", "#8FD4C1",
        "#BFE7D9", "#79C67A", "#A6D78B", "#D4E8A5", "#F0F2B6", "#FFE3A3", "#FFD3A6",
        "#FFBFA0", "#F3A6A6", "#E78E9B", "#D77CA5", "#C06BB6", "#A16BB9", "#7E7ACB",
        "#A3A3E6", "#C9C9F2", "#C1B2D6", "#E3CFE6", "#F0D9EC", "#D8D8D8", "#BFC3C9",
        "#A7ADB5", "#8E9AA4", "#73808D", "#5C6B79", "#3E4A58", "#8C6F64", "#A88C7A",
        "#C4A792", "#E0C6AF", "#F2D9C0", "#F7E6D5", "#D2ECEE", "#BCE3F5", "#A5D8FF",
        "#8FC6FF", "#74B4FF", "#5AA0F0",
    ]

    emb_adata = anndata.AnnData(adata.obsm["NLGATST_embed"])
    emb_adata.obs["pred_label"] = list(adata.obs["pred_label"])
    emb_adata.obs["pred_label"] = emb_adata.obs["pred_label"].astype("category")
    emb_adata.uns["pred_label_colors"] = list(
        islice(cycle(palette_hex), len(emb_adata.obs["pred_label"].cat.categories))
    )

    sc.pp.neighbors(emb_adata, n_neighbors=15)
    sc.tl.umap(emb_adata)
    sc.tl.paga(emb_adata, groups="pred_label")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    sc.pl.umap(emb_adata, color="pred_label", ax=axs[0], show=False, size=50, legend_fontsize=12)
    sc.pl.paga(emb_adata, color="pred_label", ax=axs[1], show=False)
    for ax, title in zip(axs, ["UMAP visualization", "PAGA inference"]):
        for side in ("right", "top", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.set_title(title, fontsize=14)

    fig.savefig(f"{args.outPath}/umap_paga.pdf", dpi=300)
    fig.savefig(f"{args.outPath}/umap_paga.png", dpi=300)
    plt.close(fig)


def Noise_Cross_Entropy(emb, adj):
    """Contrast adjacent and non-adjacent node pairs in embedding space."""
    sim_exp = torch.exp(cosine_sim_tensor(emb))
    negative = torch.mul(sim_exp, 1 - adj).sum(axis=1)
    positive = torch.mul(sim_exp, adj).sum(axis=1)
    return -torch.log(torch.div(positive + 1e-8, negative + 1e-8)).mean()


def cosine_sim_tensor(emb):
    """Compute pairwise cosine similarity with numerical safeguards."""
    dot = torch.matmul(emb, emb.T)
    length = torch.norm(emb, p=2, dim=1)
    norm = torch.matmul(length.reshape((emb.shape[0], 1)), length.reshape((emb.shape[0], 1)).T) - 5e-12
    sim = torch.div(dot, norm)
    if torch.any(torch.isnan(sim)):
        sim = torch.where(torch.isnan(sim), torch.full_like(sim, 0.4868), sim)
    return sim


def graph_alpha(spatial_locs, n_neighbors):
    """Construct an alpha-complex spatial graph from spot coordinates."""
    knn = kneighbors_graph(spatial_locs, n_neighbors=n_neighbors, mode="distance")
    graph_cut = knn.sum() / float(knn.count_nonzero())
    spatial_locs_list = spatial_locs.tolist()
    n_node = len(spatial_locs_list)

    alpha_complex = gudhi.AlphaComplex(points=spatial_locs_list)
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=graph_cut ** 2)
    initial_graph = nx.Graph()
    initial_graph.add_nodes_from(range(n_node))
    for simplex in simplex_tree.get_skeleton(1):
        if len(simplex[0]) == 2:
            initial_graph.add_edge(simplex[0][0], simplex[0][1])

    initial_graph.remove_edges_from(nx.selfloop_edges(initial_graph))
    return nx.to_scipy_sparse_array(initial_graph, format="csr")


def mclust_R(adata, num_cluster, modelNames="EEE", used_obsm="emb_pca", random_seed=0):
    """Cluster embeddings with the R package mclust."""
    np.random.seed(random_seed)
    try:
        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri as numpy2ri
    except ValueError as exc:
        raise RuntimeError(
            "rpy2 cannot locate R_HOME. Pass --r_home /path/to/R, or export R_HOME "
            "before running this script. Example: --r_home /mnt/.../user/anaconda3/envs/NP1/lib/R"
        ) from exc

    robjects.r.library("mclust")
    numpy2ri.activate()
    robjects.r["set.seed"](random_seed)
    res = robjects.r["Mclust"](numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    labels = np.array(res[-2])

    adata.obs[used_obsm] = labels
    adata.obs[used_obsm] = adata.obs[used_obsm].astype("int").astype("category")
    return adata


def fix_seed(seed):
    """Set deterministic random seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def ICC1_1(data: pd.DataFrame):
    """Compute the one-way random-effects intraclass correlation ICC(1,1)."""
    data = data.dropna()
    n, k = data.shape
    if n <= 1 or k <= 1:
        return np.nan

    grand_mean = data.values.mean()
    if np.isnan(grand_mean):
        return np.nan

    mean_per_target = data.mean(axis=1)
    mean_per_rater = data.mean(axis=0)
    sst = ((data - grand_mean) ** 2).sum().sum()
    ssb = (k * ((mean_per_target - grand_mean) ** 2)).sum()
    ssr = (n * ((mean_per_rater - grand_mean) ** 2)).sum()
    sse = sst - ssb - ssr
    msb = ssb / (n - 1)
    mse = sse / ((n - 1) * (k - 1))
    if np.isnan(msb) or np.isnan(mse):
        return np.nan
    return (msb - mse) / (msb + (k - 1) * mse)


def prepare_figure(rsz=4.0, csz=4.0, wspace=0.4, hspace=0.5, left=0.125, right=0.9, bottom=0.1, top=0.9):
    """Create a single-axis Matplotlib figure."""
    fig, axs = plt.subplots(1, 1, figsize=(csz, rsz))
    plt.subplots_adjust(wspace=wspace, hspace=hspace, left=left, right=right, bottom=bottom, top=top)
    return fig, axs


def pseudo_Spatiotemporal_Map(adata_all, pSM_values_save_filepath="./pSM_values.tsv", n_neighbors=20, resolution=1.0):
    """Estimate diffusion pseudotime from the learned embedding."""
    if "NLGATST_embed" not in adata_all.obsm:
        print("No embedding found; run training before pseudo-spatiotemporal mapping.")
        return

    print("Performing pseudo-spatiotemporal mapping...")
    adata = anndata.AnnData(adata_all.obsm["NLGATST_embed"])
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X")
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution)
    sc.tl.paga(adata)

    max_cell_for_subsampling = 5000
    if adata.shape[0] < max_cell_for_subsampling:
        sub_adata_x = adata.X
    else:
        indices = np.arange(adata.shape[0])
        sub_adata_x = adata.X[np.random.choice(indices, max_cell_for_subsampling, False), :]

    adata.uns["iroot"] = np.argmax(distance_matrix(sub_adata_x, sub_adata_x).sum(axis=1))
    sc.tl.diffmap(adata)
    sc.tl.dpt(adata)
    psm_values = adata.obs["dpt_pseudotime"].to_numpy()

    save_dir = os.path.dirname(pSM_values_save_filepath)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    np.savetxt(pSM_values_save_filepath, psm_values, fmt="%.5f", comments="")
    adata_all.obsm["pSM_values"] = psm_values
    print(f"Pseudo-spatiotemporal values saved to {pSM_values_save_filepath}.")


def plot_pSM(
    adata,
    pSM_figure_save_filepath="./pseudo-Spatiotemporal-Map.pdf",
    colormap="summer",
    scatter_sz=1.0,
    rsz=4.0,
    csz=4.0,
    wspace=0.4,
    hspace=0.5,
    left=0.125,
    right=0.9,
    bottom=0.1,
    top=0.9,
):
    """Plot pseudo-spatiotemporal values in tissue coordinates."""
    if "pSM_values" not in adata.obsm:
        print("No pseudo-spatiotemporal map found.")
        return

    fig, ax = prepare_figure(
        rsz=rsz, csz=csz, wspace=wspace, hspace=hspace, left=left, right=right, bottom=bottom, top=top
    )
    x, y = adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1]
    scatter = ax.scatter(x, y, s=scatter_sz, c=adata.obsm["pSM_values"], cmap=colormap, marker=".")
    ax.invert_yaxis()
    colorbar = fig.colorbar(scatter)
    colorbar.ax.set_ylabel("Pseudotime", labelpad=10, rotation=270, fontsize=10, weight="bold")
    ax.set_title("Pseudo-spatiotemporal map", fontsize=14)
    ax.set_facecolor("none")

    save_dir = os.path.dirname(pSM_figure_save_filepath)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(pSM_figure_save_filepath, dpi=300)
    plt.close(fig)
    print(f"Pseudo-spatiotemporal figure saved to {pSM_figure_save_filepath}.")
