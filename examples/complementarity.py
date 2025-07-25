"""
Complementarity Analysis for Graph Datasets using RINGS Framework

This script demonstrates how to:
1. Load a graph dataset from PyTorch Geometric
2. Apply various RINGS perturbations to the graph data
3. Compute complementarity metrics

Usage (from root directory):
    python -m examples.complementarity --dataset MUTAG --perturbation original

For more options:
    python -m examples.complementarity --help
"""

import numpy as np
import torch
import argparse
import time
import geoopt
from datasets.data_utils import GraphDatasetLoader 
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from rings.complementarity import ComplementarityFunctor, MatrixNormComparator
from rings.perturbations import *


def get_available_perturbations():
    """
    Return a list of available perturbation names with descriptions.

    Returns:
        dict: Mapping from perturbation name to description
    """
    return {
        "original": "No perturbation (original graph and features)",
        # Node feature perturbations
        "empty-features": "Replace node features with empty (zero) vectors",
        "random-features": "Replace node features with random values",
        # Graph structure perturbations
        "empty-graph": "Remove all edges from the graph",
        "complete-graph": "Create edges between all pairs of nodes",
        "random-graph": "Generate a random graph structure",
    }


def create_perturbation(name, seed=42):
    """
    Create a perturbation transform based on the given name and seed.

    Args:
        name (str): Name of the perturbation
        seed (int): Random seed for reproducible perturbations

    Returns:
        BaseTransform: A transform that can be applied to graph data
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    if name == "original":
        return Original()

    # Node feature perturbations
    elif name == "empty-features":
        return EmptyFeatures()
    elif name == "random-features":
        return RandomFeatures(fixed_dimension=None)

    # Graph structure perturbations
    elif name == "empty-graph":
        return EmptyGraph()
    elif name == "complete-graph":
        return CompleteGraph()
    elif name == "random-graph":
        return RandomGraph(shuffle=True)

    else:
        raise ValueError(f"Unknown perturbation: {name}")


def apply_perturbation(dataset, perturbation):
    """
    Apply a perturbation to all graphs in a dataset.

    Args:
        dataset: PyTorch Geometric dataset
        perturbation: Transform to apply

    Returns:
        list: List of transformed graph data objects
    """
    print(f"Applying perturbation: {perturbation.__class__.__name__}")
    transformed_dataset = [perturbation(data.clone()) for data in dataset]
    return transformed_dataset


def create_complementarity_functor(n_jobs=1, **kwargs):
    """
    Create a ComplementarityFunctor with standard parameters.

    The ComplementarityFunctor measures the difference between:
    1. The metric space of node features (using euclidean distance)
    2. The metric space of graph structure (using shortest path distance)

    Args:
        n_jobs (int): Number of parallel jobs (-1 for all cores)

    Returns:
        ComplementarityFunctor: Configured functor
    """
    return ComplementarityFunctor(
        # Metric for node features (pairwise Euclidean distance)
        feature_metric= kwargs.get("feature_metric", "euclidean"),
        # Metric for graph structure (shortest path distance between nodes)
        graph_metric="shortest_path_distance",
        # Method to compare the two metric spaces
        comparator=MatrixNormComparator,
        # Parallelization parameter
        n_jobs=n_jobs,
        # Norm used by the comparator
        norm="L11",
        # Manifold type for node features (if applicable)
        manifold=kwargs.get("manifold", None),
    )


def compute_complementarity(dataloader, functor):
    """
    Compute complementarity scores for all graphs in the dataloader.

    Args:
        dataloader: PyTorch Geometric dataloader containing graphs
        functor: ComplementarityFunctor to compute scores

    Returns:
        numpy.ndarray: Array of complementarity scores
    """
    start_time = time.time()
    print("Computing complementarity scores...")

    all_scores = []
    for batch in dataloader:
        # The functor computes complementarity for each graph in the batch
        results = functor(batch)

        # Extract the complementarity scores from the results
        batch_scores = results["complementarity"]
        all_scores.extend(batch_scores.tolist())

    duration = time.time() - start_time
    print(f"Computation completed in {duration:.2f} seconds")

    return np.array(all_scores)


def main():
    """Main function to run the complementarity analysis.

    User inputs are handled via command line arguments:
        --perturbation: "Perturbation to apply to the dataset (default: original)"
        --dataset: "Name of the TU dataset to use (default: MUTAG)"
        --seed: "Random seed for reproducibility (default: 42)"
        --batch-size: "Batch size for dataloader (default: 32)"
        --n-jobs: "Number of parallel jobs (-1 for all cores, default: 1)"
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Compute complementarity metrics for graph datasets"
    )

    # Get available perturbations for help text
    perturbations = get_available_perturbations()
    perturbation_help = "Available perturbations:\n" + "\n".join(
        f"  {name}: {desc}" for name, desc in perturbations.items()
    )

    parser.add_argument(
        "--perturbation",
        type=str,
        default="original",
        choices=list(perturbations.keys()),
        help=f"Perturbation to apply to the dataset. {perturbation_help}",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="citeseer",
        help="Dataset name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for dataloader",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (-1 for all cores)",
    )
    parser.add_argument(
        "--manifold",
        type=str,
        default="lorentz",
        help="Type of manifold (default:euclidean, options: 'euclidean', 'lorentz', 'poincare')",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Curvature (default: 1.0)",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load dataset
    print(f"Loading {args.dataset} dataset...")
    loader = GraphDatasetLoader(
        dataset_name=args.dataset
    )
    dataset = loader.load()
    print(f"Type: {type(dataset)}")
    if isinstance(dataset, list):
        print(f"Type of first item: {type(dataset[0])}")
        print(f"Data object: {dataset[0]}")
    else:
        print("Dataset is a single graph.")
        print(f"Data object: {dataset}")
        # Wrap single graph in a list for uniform processing downstream
        dataset = [dataset]

    # 2. Map features in the manifold space
    if args.manifold == "euclidean":
        manifold = geoopt.Euclidean()
        feature_metric = "euclidean"
    elif args.manifold == "lorentz":
        manifold = geoopt.Lorentz(k=args.c)
        feature_metric = "custom_manifold"
    elif args.manifold == "poincare":
        manifold = geoopt.PoincareBall(c=args.c)
        feature_metric = "custom_manifold"
    else:
        raise ValueError(f"Unknown manifold type: {args.manifold}")
    
    # 3. Project node features to manifold
    for data_item in dataset:
        if isinstance(manifold, geoopt.Euclidean):
            data_item.x = data_item.x.double() 
        elif isinstance(manifold, geoopt.Lorentz) or isinstance(manifold, geoopt.PoincareBall):   
            print("Shape before projection",data_item.x.shape)  
            data_item.x = data_item.x.double() # Ensure features are in double precision
            print("Node feature:", data_item.x)
            if isinstance(manifold, geoopt.Lorentz):
                zeros = torch.zeros(data_item.x.size(0), data_item.x.size(1) - 1, dtype=torch.float64)  # Zero vector for time component
                ones = torch.ones(data_item.x.size(0), 1, dtype=torch.float64)  # One vector for time component
                origin = torch.cat([ones, zeros], dim=1)  # Origin of the space
                tangent_vector = manifold.proju(origin, data_item.x)  # Project features to tangent space at origin
                data_item.x = manifold.projx(manifold.expmap(origin, tangent_vector))  # Projection to manifold
                print("Shape after projection",data_item.x.shape) 
                print(f"Data object after projection: {dataset[0]}")
            elif isinstance(manifold, geoopt.PoincareBall):
                origin = torch.zeros(data_item.x.size(0), data_item.x.size(1), dtype=torch.float64)
                data_item.x = manifold.projx(manifold.expmap(origin, data_item.x))  # Projection to manifold
                print("Shape after projection",data_item.x.shape) 
                print("Node feature:", data_item.x)
                
        # elif isinstance(manifold, geoopt.Lorentz) or isinstance(manifold, geoopt.PoincareBall):   
        #     print("Shape before projection",data_item.x.shape)  
        #     data_item.x = data_item.x.double()  # Ensure features are in double precision
        #     origin = torch.zeros(data_item.x.size(0), data_item.x.size(1) + 1, dtype=torch.float64) # Origin of the space
        #     zero_time_component = torch.zeros((data_item.x.size(0), 1))  # Zero time component to ensure the point live in the tangent space of the origin
        #     tangent_vector = torch.cat((zero_time_component, data_item.x), dim=1)  # Node features in tangent space of the origin
        #     data_item.x = manifold.expmap(origin, tangent_vector)  # Projection to manifold
        #     print("Shape after projection",data_item.x.shape) 
        #     print(f"Data object after projection: {dataset[0]}")



    # 4. Create perturbation and apply to dataset
    perturbation = create_perturbation(args.perturbation, args.seed)
    transformed_dataset = apply_perturbation(dataset, perturbation)
    print(f"Transformed dataset size: {len(transformed_dataset)}")

    # 5. Create dataloader for batch processing
    dataloader = DataLoader(
        transformed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # 6. Create complementarity functor
    functor = create_complementarity_functor(feature_metric=feature_metric, n_jobs=args.n_jobs, manifold=manifold, c=args.c)

    # 7. Compute complementarity scores
    scores = compute_complementarity(dataloader, functor)

    # 8. Calculate and print statisti
    mean_score = np.mean(scores)
    std_score = np.std(scores)


    output_file = f"/home/dionisia/benchmark_evaluation/results/complementarity_scores_{args.dataset}_{args.manifold}_{args.c}.txt"

    summary = (
        "\n" + "=" * 60 + "\n" +
        f"{'📊 Results Summary'.center(60)}\n" +
        "=" * 60 + "\n" +
        f"📁 Dataset:         {args.dataset}\n" +
        f"🧪 Perturbation:    {args.perturbation}\n" +
        "-" * 60 + "\n" +
        f"✅ Mean Complementarity:  {mean_score:.4f}\n" +
        f"📉 Std. Deviation:        {std_score:.4f}\n" +
        f"📈 Number of Graphs:      {len(scores)}\n" +
        "=" * 60 + "\n\n"
    )

    # Print to console
    print(summary)

    # Save to file
    with open(output_file, "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
