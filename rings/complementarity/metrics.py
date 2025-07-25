"Methods for lifting attributes and graphs to metric spaces."

import sklearn
import warnings

from sklearn.metrics.pairwise import pairwise_distances as skl_pairwise_distances
from sklearn.metrics.pairwise import _VALID_METRICS

import torch
import numpy as np
import networkx as nx

import rings.complementarity.metrics as supported_metrics

#  ╭──────────────────────────────────────────────────────────╮
#  │ Lifters                                                  │
#  ╰──────────────────────────────────────────────────────────╯


def lift_attributes(X, metric, n_jobs, **kwargs):
    """
    Lift node attributes to a metric space.

    This function transforms node attributes into a pairwise distance matrix,
    effectively "lifting" them into a metric space. It supports standard metrics
    from scikit-learn as well as custom metrics defined in this module.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Node feature matrix where each row represents a node's features.
    metric : str
        Name of the metric to use. Can be any metric supported by scikit-learn's
        pairwise_distances or a custom metric defined in this module.
    n_jobs : int
        Number of jobs to run in parallel for distance computation.
        -1 means using all processors.
    **kwargs : dict
        Additional keyword arguments to pass to the metric function.

    Returns
    -------
    ndarray of shape (n_samples, n_samples)
        Distance matrix representing pairwise distances between node features.

    Raises
    ------
    RuntimeError
        If the specified metric is not supported.

    Examples
    --------
    >>> import numpy as np
    >>> from rings.complementarity.metrics import lift_attributes
    >>>
    >>> # Node feature matrix: 3 nodes with 2 features each
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>>
    >>> # Using scikit-learn's euclidean metric
    >>> D_euclidean = lift_attributes(X, metric="euclidean", n_jobs=1)
    >>> print(D_euclidean)
    [[0.         2.82842712 5.65685425]
     [2.82842712 0.         2.82842712]
     [5.65685425 2.82842712 0.        ]]
    >>>
    >>> # Using custom metric
    >>> D_custom = lift_attributes(X, metric="standard_feature_metrics", n_jobs=1, metric="manhattan")
    >>> print(D_custom)
    [[0. 4. 8.]
     [4. 0. 4.]
     [8. 4. 0.]]
    """
    operator = getattr(supported_metrics, metric, None)

    if operator is None:
        operator = generalized_feature_metrics

    return operator(X, metric=metric, n_jobs=n_jobs, **kwargs)


def lift_graph(G, metric, **kwargs):
    """
    Lift graph structure to a metric space.

    This function transforms a graph structure into a pairwise distance matrix
    using a specified graph metric, effectively "lifting" the graph into a
    metric space.

    Parameters
    ----------
    G : networkx.Graph
        The input graph whose structure will be lifted to a metric space.
    metric : str
        Name of the graph metric to use (e.g., "shortest_path_distance",
        "diffusion_distance", "resistance_distance").
    **kwargs : dict
        Additional keyword arguments to pass to the metric function.

    Returns
    -------
    ndarray of shape (n_nodes, n_nodes)
        Distance matrix representing pairwise distances between nodes
        according to the specified graph metric.

    Raises
    ------
    RuntimeError
        If the specified metric is not supported.

    Examples
    --------
    >>> import networkx as nx
    >>> from rings.complementarity.metrics import lift_graph
    >>>
    >>> # Create a simple graph: path graph with 3 nodes (0-1-2)
    >>> G = nx.path_graph(3)
    >>>
    >>> # Lift using shortest path distance
    >>> D_sp = lift_graph(G, metric="shortest_path_distance")
    >>> print(D_sp)
    [[0. 1. 2.]
     [1. 0. 1.]
     [2. 1. 0.]]
    >>>
    >>> # Lift using diffusion distance
    >>> D_diff = lift_graph(G, metric="diffusion_distance", num_steps=1)
    >>> # Result will be a diffusion-based distance matrix
    """
    # Immediately return zero metric space if graph is empty.
    if G.number_of_edges() == 0:
        n = G.number_of_nodes()
        return np.zeros((n, n))

    operator = getattr(supported_metrics, metric, None)

    if operator is None:
        raise RuntimeError(f"Unsupported metric: {metric}")

    return operator(G, **kwargs)


#  ╭──────────────────────────────────────────────────────────╮
#  │ Attribute Metrics                                        │
#  ╰──────────────────────────────────────────────────────────╯


def standard_feature_metrics(X, **kwargs):
    """
    Calculate pairwise distances between node features.

    This function calculates a distance matrix between node features using
    scikit-learn's pairwise_distances function. All metrics supported by
    scikit-learn can be used.

    Parameters
    ----------
    X : array_like of shape (n_samples, n_features)
        Node feature matrix, with rows indicating nodes and columns
        corresponding to feature dimensions.
    **kwargs : dict
        Additional keyword arguments including:

        - metric : str, default="euclidean"
            The metric to use for distance calculation. Options include:
            "euclidean", "manhattan", "cosine", "jaccard", etc.
        - n_jobs : int, default=None
            Number of jobs to run in parallel. None means 1 unless in a
            joblib.parallel_backend context.

    Returns
    -------
    ndarray of shape (n_samples, n_samples)
        Matrix of pairwise distances between node features.

    Examples
    --------
    >>> import numpy as np
    >>> from rings.complementarity.metrics import standard_feature_metrics
    >>>
    >>> # Node features for 3 nodes
    >>> X = np.array([[1, 0], [0, 1], [1, 1]])
    >>>
    >>> # Calculate Euclidean distances
    >>> D_eucl = standard_feature_metrics(X, metric="euclidean")
    >>> print(D_eucl)
    [[0.         1.41421356 1.        ]
     [1.41421356 0.         1.        ]
     [1.         1.         0.        ]]
    >>>
    >>> # Calculate Manhattan distances
    >>> D_manh = standard_feature_metrics(X, metric="manhattan")
    >>> print(D_manh)
    [[0. 2. 1.]
     [2. 0. 1.]
     [1. 1. 0.]]
    """
    return skl_pairwise_distances(
        X,
        metric=kwargs.get("metric", "euclidean"),
        n_jobs=kwargs.get("n_jobs", None),
    )

def generalized_feature_metrics(X, **kwargs):
    """
    Calculate pairwise distances between node features.
    Compute pairwise distances either via sklearn metrics or a custom manifold.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Input data.
    metric : str
        If metric == "custom_manifold", use `manifold.dist()`.
        Else, fallback to sklearn metrics.
    manifold : object, optional
        Manifold object with a  (e.g., geoopt.Lorentz) `.dist()` method, required if metric == "custom_manifold".
    **kwargs :
        Extra keyword arguments.

    Returns
    -------
    Distance matrix
    """
    
    metric = kwargs.get("metric", "euclidean")

    if metric == "custom_manifold":
        if not getattr(generalized_feature_metrics, "_printed", False):
            print("Computing distances using custom manifold...")
            generalized_feature_metrics._printed = True
        manifold = kwargs.get("manifold", None)
        if manifold is None:
            raise ValueError("Must provide manifold when metric='custom_manifold'")

        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
            print("X:", X.shape)

        X1 = X.unsqueeze(1)
        X2 = X.unsqueeze(0)


        dist =  manifold.dist(X1, X2).detach().cpu().numpy()

        print("Dist:", dist.shape)

        if np.isinf(dist).any():
            print("Warning: infs detected in distance matrix!")
            print("NaN locations:", np.argwhere(np.isinf(dist)))
            print("Distance matrix:\n", dist)

        return dist #manifold.dist(X1, X2).detach().cpu().numpy()



    elif metric in _VALID_METRICS:
        if not getattr(generalized_feature_metrics, "_printed", False):
            print("Computing distances using sklearn metrics...")
            generalized_feature_metrics._printed = True

        return skl_pairwise_distances(
            X, 
            metric=metric, 
            n_jobs=kwargs.get("n_jobs", None)
            )
    else:
        raise ValueError(f"Unsupported metric: {metric}")


#  ╭──────────────────────────────────────────────────────────╮
#  │ Graph Metrics                                            │
#  ╰──────────────────────────────────────────────────────────╯


def diffusion_distance(G, **kwargs):
    """
    Calculate diffusion distance between vertices of a graph.

    The diffusion distance measures how similar two nodes are in terms of their
    connectivity patterns in the graph. It is based on a diffusion process where
    heat (or probability) spreads throughout the graph over time.

    Parameters
    ----------
    G : networkx.Graph
        Input graph. All attributes of the graph will be ignored in
        the subsequent calculations.
    num_steps : int, default=1
        Number of steps for the diffusion operator that is used for
        the distance calculations.
    norm : bool, default=True
        Whether to normalize the Laplacian.
    symmetric : bool, default=True
        Whether to normalize the Laplacian symmetrically.
    n_jobs : int, optional
        Number of jobs for parallel distance computation.

    Returns
    -------
    ndarray of shape (n_nodes, n_nodes)
        Matrix of pairwise diffusion distances between nodes.

    Notes
    -----
    This implementation computes diffusion distances by:
    1. Computing the graph Laplacian matrix (normalized or unnormalized)
    2. Computing the matrix Ψ = L^t where L is the Laplacian and t is num_steps
    3. Computing pairwise Euclidean distances between rows of Ψ

    If the graph has isolated nodes, the function returns NaN.

    Examples
    --------
    >>> import networkx as nx
    >>> import numpy as np
    >>> from rings.complementarity.metrics import diffusion_distance
    >>>
    >>> # Create a simple path graph: 0-1-2-3
    >>> G = nx.path_graph(4)
    >>>
    >>> # Calculate diffusion distances
    >>> D = diffusion_distance(G, num_steps=1)
    >>>
    >>> # Verify properties of the distance matrix
    >>> np.testing.assert_almost_equal(np.diag(D), np.zeros(4))  # Zero diagonal
    >>> np.testing.assert_almost_equal(D, D.T)  # Symmetric matrix
    """
    t = kwargs.get("num_steps", 1)
    n_jobs = kwargs.get("n_jobs", None)
    symmetric = kwargs.get("symmetric", True)
    norm = kwargs.get("norm", True)

    if not norm and not symmetric:
        warnings.warn(
            "Assuming default Laplacian, which is symmetric.",  # noqa: E501
            UserWarning,
        )
        symmetric = True

    A = nx.adjacency_matrix(G).todense()
    D = np.diag(A.sum(axis=1))

    # Bail out early on if there are isolated nodes.
    if (np.diagonal(D) == 0).any():
        return np.nan

    # NEW VERSION
    L = _compute_laplacian(A, D, norm=norm, symmetric=symmetric)
    psi = _compute_psi(L, t, symmetric=symmetric)

    if np.iscomplexobj(psi):
        warnings.warn(
            "Input data contains complex numbers. The imaginary part will be discarded.",  # noqa: E501
            UserWarning,
        )
        psi = np.real(psi)
    return pairwise_distances(psi, metric="euclidean", n_jobs=n_jobs)


def heat_kernel_distance(G, **kwargs):
    """
    Calculate heat kernel distance between vertices of a graph.

    The heat kernel distance is a measure of node similarity based on the heat
    diffusion process on the graph. It captures how heat flows through the graph
    structure over time, providing a rich metric for comparing nodes.

    Parameters
    ----------
    G : networkx.Graph
        Input graph. All attributes of the graph will be ignored in
        the subsequent calculations.
    num_steps : int, default=1
        Number of time steps for the diffusion process. Controls how far
        the heat diffuses through the graph.
    norm : bool, default=True
        Whether to normalize the Laplacian.
    symmetric : bool, default=True
        Whether to normalize the Laplacian symmetrically.
    n_jobs : int, optional
        Number of jobs for parallel distance computation.

    Returns
    -------
    ndarray of shape (n_nodes, n_nodes)
        Matrix of pairwise heat kernel distances between nodes.

    Notes
    -----
    This implementation computes heat kernel distances by:
    1. Computing the graph Laplacian matrix
    2. Computing eigenvalues and eigenvectors of the Laplacian
    3. Scaling eigenvectors by e^(-t * eigenvalues)
    4. Computing pairwise Euclidean distances between rows of the scaled eigenvectors

    If complex numbers are encountered, the real part is used and a warning is issued.

    Examples
    --------
    >>> import networkx as nx
    >>> import numpy as np
    >>> from rings.complementarity.metrics import heat_kernel_distance
    >>>
    >>> # Create a simple grid graph
    >>> G = nx.grid_2d_graph(2, 2)  # 2x2 grid
    >>> G = nx.convert_node_labels_to_integers(G)  # Relabel as integers
    >>>
    >>> # Calculate heat kernel distances
    >>> D = heat_kernel_distance(G, num_steps=0.1)
    >>>
    >>> # Verify properties
    >>> assert D.shape == (4, 4)  # 4 nodes in 2x2 grid
    >>> np.testing.assert_almost_equal(np.diag(D), np.zeros(4))  # Zero diagonal
    >>> np.testing.assert_almost_equal(D, D.T)  # Symmetric matrix
    """
    t = kwargs.get("num_steps", 1)
    n_jobs = kwargs.get("n_jobs", None)
    symmetric = kwargs.get("symmetric", True)
    norm = kwargs.get("norm", True)

    A = nx.adjacency_matrix(G).todense()
    D = np.diag(A.sum(axis=1))

    L = _compute_laplacian(A, D, norm=norm, symmetric=symmetric)

    eigenvalues, eigenvectors = _compute_spectrum(L, symmetric=symmetric)
    scales = np.exp(-t * eigenvalues)
    X = scales * eigenvectors

    if np.iscomplexobj(X):
        warnings.warn(
            "Input data contains complex numbers. The imaginary part will be discarded.",  # noqa: E501
            UserWarning,
        )
        X = np.real(X)

    return pairwise_distances(X, metric="euclidean", n_jobs=n_jobs)


def resistance_distance(G, **kwargs):
    """
    Calculate resistance distance between vertices of a graph.

    The resistance distance between two nodes is the effective resistance when the graph
    is viewed as an electrical network with edges as resistors. It is also equal to the
    commute time between nodes in a random walk (up to a scaling factor).

    Parameters
    ----------
    G : networkx.Graph
        Input graph. Graph attributes are ignored in the calculations.
    weight : str or None, optional
        The edge attribute that holds the numerical value used as a
        weight. If set to `None`, the graph is treated as unweighted.
        For compatibility with other RINGS metrics, this should be
        consistent with other metrics.
    **kwargs : dict
        Additional keyword arguments. Only required for compatibility
        with the metric API.

    Returns
    -------
    ndarray of shape (n_nodes, n_nodes)
        Matrix of pairwise resistance distances between nodes.

    Notes
    -----
    This function uses NetworkX's built-in resistance_distance function.
    If the graph is not connected, the function will raise a NetworkXError,
    which this wrapper catches and returns NaN.

    For unweighted graphs, the resistance distance is directly related to
    the commute time and the pseudo-inverse of the graph Laplacian.

    Examples
    --------
    >>> import networkx as nx
    >>> import numpy as np
    >>> from rings.complementarity.metrics import resistance_distance
    >>>
    >>> # Create a simple path graph: 0-1-2-3
    >>> G = nx.path_graph(4)
    >>>
    >>> # Calculate resistance distances
    >>> D = resistance_distance(G)
    >>>
    >>> # For a path graph, resistance distance equals shortest path distance
    >>> sp_distance = nx.floyd_warshall_numpy(G)
    >>> np.testing.assert_almost_equal(D, sp_distance)
    >>>
    >>> # Verify that diagonal elements are zero
    >>> np.testing.assert_almost_equal(np.diag(D), np.zeros(4))
    """
    try:
        distances = nx.resistance_distance(G)
        distances = nx.utils.dict_to_numpy_array(distances)
    except nx.NetworkXError:
        distances = [np.nan]

    return distances


def shortest_path_distance(G, **kwargs):
    """
    Calculate shortest-path distance between all pairs of vertices in a graph.

    This function computes the shortest path lengths between all pairs of nodes
    in the graph using the Floyd-Warshall algorithm. The result is a distance
    matrix where each element [i,j] represents the shortest path distance from
    node i to node j.

    Parameters
    ----------
    G : networkx.Graph
        Input graph.
    weight : str or None, optional
        The edge attribute that holds the numerical value used as a
        weight. If set to `None` (default), the graph is treated as unweighted
        and each edge has weight 1.
    **kwargs : dict
        Additional keyword arguments for compatibility with the metric API.
        These are not used by this function but allow for consistent interfaces.

    Returns
    -------
    ndarray of shape (n_nodes, n_nodes)
        Matrix of shortest path distances between all pairs of nodes.

    Notes
    -----
    The time complexity of the Floyd-Warshall algorithm is O(n³), where n is the
    number of nodes in the graph. For very large graphs, consider using other
    algorithms such as Dijkstra's algorithm on-demand for specific node pairs.

    Examples
    --------
    >>> import networkx as nx
    >>> import numpy as np
    >>> from rings.complementarity.metrics import shortest_path_distance
    >>>
    >>> # Create an unweighted graph
    >>> G1 = nx.Graph()
    >>> G1.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)])  # Pentagon
    >>> D1 = shortest_path_distance(G1)
    >>> print(D1)  # Each entry [i,j] is the shortest path length from i to j
    [[0. 1. 2. 2. 1.]
     [1. 0. 1. 2. 2.]
     [2. 1. 0. 1. 2.]
     [2. 2. 1. 0. 1.]
     [1. 2. 2. 1. 0.]]
    >>>
    >>> # Create a weighted graph
    >>> G2 = nx.Graph()
    >>> G2.add_weighted_edges_from([(0, 1, 1), (1, 2, 3), (0, 2, 5)])  # Triangle
    >>> D2 = shortest_path_distance(G2, weight='weight')
    >>> # Path 0->2 should use the direct edge, not 0->1->2, due to weights
    >>> assert D2[0, 2] == 4  # Shortest path is 0->1->2 with weight 1+3=4
    """
    return nx.floyd_warshall_numpy(G)


#  ╭──────────────────────────────────────────────────────────╮
#  │ Spectral Helper Functions                                │
#  ╰──────────────────────────────────────────────────────────╯


def _compute_laplacian(A, D, norm=True, symmetric=True):
    """
    Compute the Laplacian matrix of a graph.

    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix of the graph.
    D : numpy.ndarray
        Degree matrix of the graph.
    norm : bool, default=True
        If True, compute the normalized Laplacian.
    symmetric : bool, default=True
        If True, compute the symmetric normalized Laplacian.

    Returns
    -------
    L : numpy.ndarray
        The Laplacian matrix of the graph.

    Notes
    -----
    The Laplacian matrix is defined as L = D - A, where D is the degree matrix
    and A is the adjacency matrix.

    If `norm` is True and `symmetric` is True, the symmetric normalized Laplacian
    is computed as L = I - D^(-1/2) * A * D^(-1/2).

    If `norm` is True and `symmetric` is False, the random walk normalized Laplacian
    is computed as L = I - D^(-1) * A.

    Examples
    --------
    >>> import numpy as np
    >>> from rings.complementarity.metrics import _compute_laplacian
    >>>
    >>> # Create a path graph adjacency matrix
    >>> A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> D = np.diag(np.sum(A, axis=1))  # Degree matrix
    >>>
    >>> # Compute unnormalized Laplacian
    >>> L1 = _compute_laplacian(A, D, norm=False, symmetric=True)
    >>> print(L1)
    [[ 1 -1  0]
     [-1  2 -1]
     [ 0 -1  1]]
    >>>
    >>> # Compute symmetric normalized Laplacian
    >>> L2 = _compute_laplacian(A, D, norm=True, symmetric=True)
    >>> print(L2)
    [[ 1.         -0.70710678  0.        ]
     [-0.70710678  1.         -0.70710678]
     [ 0.         -0.70710678  1.        ]]
    """
    if norm and symmetric:
        D_inv_sqrt = np.diag(1.0 / np.sqrt(D.diagonal()))
        L = np.eye(*A.shape) - D_inv_sqrt @ A @ D_inv_sqrt
    elif norm and not symmetric:
        L = np.eye(*A.shape) - np.diag(1.0 / D.diagonal()) @ A
    else:
        L = D - A
    return L


def _compute_psi(L, t, symmetric=True):
    """
    Compute Psi matrix for diffusion distance calculation.

    This function computes a matrix Ψ by raising the Laplacian eigenvalues to power t
    and multiplying with the eigenvectors. This represents the state of a diffusion
    process after time t.

    Parameters
    ----------
    L : array_like
        The Laplacian matrix of the graph.
    t : float
        The time parameter for the diffusion process, controlling how far the
        diffusion spreads. Larger values mean more diffusion.
    symmetric : bool, default=True
        If True, assumes the Laplacian matrix is symmetric and uses specialized
        eigendecomposition for symmetric matrices.

    Returns
    -------
    psi : ndarray
        The Psi matrix computed from the eigenvalues and eigenvectors of the Laplacian.

    Examples
    --------
    >>> import numpy as np
    >>> from rings.complementarity.metrics import _compute_psi, _compute_laplacian
    >>>
    >>> # Create a simple Laplacian matrix
    >>> A = np.array([[0, 1], [1, 0]])
    >>> D = np.diag([1, 1])
    >>> L = _compute_laplacian(A, D, norm=True, symmetric=True)
    >>>
    >>> # Compute Psi for different time steps
    >>> psi_small_t = _compute_psi(L, t=0.1)
    >>> psi_large_t = _compute_psi(L, t=1.0)
    >>>
    >>> # As t increases, diffusion spreads further
    >>> print(f"Psi(t=0.1): {psi_small_t}")
    >>> print(f"Psi(t=1.0): {psi_large_t}")
    """
    eigenvalues, eigenvectors = _compute_spectrum(L, symmetric=symmetric)
    eigenvalues = np.power(eigenvalues, t)
    psi = eigenvalues * eigenvectors
    return psi


def _compute_spectrum(L, symmetric=True):
    """
    Compute the eigenvalues and eigenvectors of the Laplacian matrix.

    The spectrum (eigenvalues and eigenvectors) of the Laplacian matrix provides
    important information about the graph structure and is used in many spectral
    graph theory applications.

    Parameters
    ----------
    L : array_like
        The Laplacian matrix of the graph.
    symmetric : bool, default=True
        If True, assumes the Laplacian matrix is symmetric and uses specialized
        eigendecomposition for symmetric matrices, which is faster and more stable.

    Returns
    -------
    eigenvalues : ndarray
        The eigenvalues of the Laplacian matrix.
    eigenvectors : ndarray
        The eigenvectors of the Laplacian matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from rings.complementarity.metrics import _compute_spectrum
    >>>
    >>> # Create a simple Laplacian matrix for a path on 3 vertices
    >>> L = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])
    >>>
    >>> # Compute spectrum
    >>> eigenvalues, eigenvectors = _compute_spectrum(L)
    >>> print(f"Eigenvalues: {eigenvalues}")
    >>> print(f"First eigenvector: {eigenvectors[:, 0]}")

    Notes
    -----
    For a connected graph with n nodes, the eigenvalues of the Laplacian satisfy:
    0 = λ₁ < λ₂ ≤ ... ≤ λₙ

    The multiplicity of the eigenvalue 0 equals the number of connected components.
    """
    if symmetric:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    else:
        eigenvalues, eigenvectors = np.linalg.eig(L)
    return eigenvalues, eigenvectors
