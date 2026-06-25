"""Standalone preprocessing utilities for spatial transcriptomics data."""

from __future__ import annotations

import gudhi
import networkx as nx
import scanpy as sc
from sklearn.neighbors import kneighbors_graph


def preprocessing_data(adata, n_top_genes):
    """Apply gene filtering, library-size normalization, log transform, and HVG selection."""
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4, exclude_highly_expressed=True, inplace=False)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_top_genes)
    return adata[:, adata.var.highly_variable].copy()


def preprocessing_stereo_data(adata, n_top_genes):
    """Preprocess Stereo-seq-like data with stricter cell and gene filters."""
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_counts=15)
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.normalize_total(adata, target_sum=1e4, exclude_highly_expressed=True, inplace=False)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_top_genes)
    return adata[:, adata.var.highly_variable].copy()


def graph_alpha(spatial_locs, n_neighbors):
    """Construct an alpha-complex spatial proximity graph."""
    knn = kneighbors_graph(spatial_locs, n_neighbors=n_neighbors, mode="distance")
    graph_cut = knn.sum() / float(knn.count_nonzero())
    alpha_complex = gudhi.AlphaComplex(points=spatial_locs.tolist())
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=graph_cut ** 2)

    graph = nx.Graph()
    graph.add_nodes_from(range(len(spatial_locs)))
    for simplex in simplex_tree.get_skeleton(1):
        if len(simplex[0]) == 2:
            graph.add_edge(simplex[0][0], simplex[0][1])
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return nx.to_scipy_sparse_array(graph, format="csr")
