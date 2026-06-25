"""Command-line configuration for NLGATST experiments."""

from __future__ import annotations

import argparse


def parameter_setting() -> argparse.ArgumentParser:
    """Create the argument parser used by the training entry point."""
    parser = argparse.ArgumentParser(
        description="Nonlinear graph attention for spatial transcriptomics clustering."
    )

    parser.add_argument("--platform", type=str, default="Visium", choices=["Visium", "starmap", "h5ad"])
    parser.add_argument("--label", type=str, default="T", choices=["T", "F"])
    parser.add_argument("--image", type=str, default="T", choices=["T", "F"])
    parser.add_argument("--basePath", "-bp", type=str, default="Datasets/151507")
    parser.add_argument("--inputPath", "-IP", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="151507")
    parser.add_argument("--ground_truth_dir", type=str, default="Datasets/SpatialDE_clustering")
    parser.add_argument("--n_clusters", type=int, default=7)
    parser.add_argument("--n_top_genes", type=int, default=2000)
    parser.add_argument("--gpu_id", type=str, default="1", help="Visible CUDA device id, e.g. 0 or 2.")
    parser.add_argument(
        "--r_home",
        type=str,
        default="/mnt/402f169c-136d-4c4c-b598-a9732ee96752/user/anaconda3/envs/NP1/lib/R",
        help="R installation path required by rpy2/mclust.",
    )

    parser.add_argument("--use_cuda", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_1", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--head", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--inner_steps", type=int, default=3)
    parser.add_argument("--scale_factor", type=float, default=2.0)
    parser.add_argument("--margin", type=float, default=1.0)

    parser.add_argument("--aggtype", type=str, default="Softmax", choices=["Softmax"])
    parser.add_argument("--we1", type=float, default=0.2, help="Adjacency reconstruction loss weight.")
    parser.add_argument("--we2", type=float, default=1.0, help="Contrastive loss weight.")
    parser.add_argument("--we3", type=float, default=2.0, help="Feature reconstruction loss weight.")
    return parser
