# NLGATST

NLGATST is a nonlinear graph attention workflow for spatial transcriptomics clustering. The current implementation trains a softmax-based nonlinear GAT encoder on spatially defined graphs, clusters the learned embedding with `mclust`, and exports spatial domain plots and class-level expression summaries.

## Repository Layout

```text
.
|-- run.py                    # Main training and analysis entry point
|-- utilities.py              # Command-line arguments
|-- NonlinearGAT/             # Model, training loop, and analysis utilities
|-- preprocess/               # Standalone preprocessing helpers
|-- requirements.txt          # Python package requirements
`-- .gitignore                # Local data, outputs, and environment exclusions
```

Large datasets and generated outputs are intentionally excluded from version control. Keep them under `Datasets/` or an external data directory.

## Usage

Install the Python dependencies and make sure R has the `mclust` package available for clustering.

```bash
pip install -r requirements.txt
R -e "install.packages('mclust')"
```

Run the default Visium example:

```bash
python run.py --basePath Datasets/151507 --dataset_name 151507 --n_clusters 7
```

Outputs are written to `<basePath>_NLGATST/`, including clustering labels, spatial figures, UMAP/PAGA plots, and the final AnnData object.

## Test data

A small example dataset or instructions for downloading the test dataset are provided in the Datasets directory. Users can run the example using:

python run.py --basePath Datasets/starmap --dataset_name starmap --n_clusters 7

The input data should contain the spatial expression matrix and spatial coordinate information. The outputs include clustering labels, spatial domain plots, UMAP/PAGA plots, and the final AnnData object.