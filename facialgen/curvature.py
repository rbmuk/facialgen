from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import lstsq, solve
from scipy.optimize import linprog
from scipy.sparse.csgraph import connected_components, shortest_path


def largest_connected_component(A: sp.spmatrix) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Extract the largest connected component of an undirected sparse graph.

    Parameters
    ----------
    A : sp.spmatrix
        Sparse adjacency matrix of shape (n, n).

    Returns
    -------
    A_lcc : sp.csr_matrix
        Adjacency matrix of the largest connected component.
    nodes : np.ndarray
        Original node indices kept in the largest connected component.
    """
    A = sp.csr_matrix(A)
    n_components, labels = connected_components(A, directed=False, connection="weak")

    if n_components == 1:
        nodes = np.arange(A.shape[0])
        return A, nodes

    counts = np.bincount(labels)
    lcc_label = np.argmax(counts)
    nodes = np.where(labels == lcc_label)[0]
    A_lcc = A[nodes][:, nodes].tocsr()
    return A_lcc, nodes


def distance_matrix_from_adjacency(
    A: sp.spmatrix,
    *,
    check_connected: bool = True,
    unweighted: bool = True,
    directed: bool = False,
    return_predecessors: bool = False,
) -> np.ndarray:
    """
    Compute the all-pairs shortest path distance matrix D.

    Parameters
    ----------
    A : sp.spmatrix
        Sparse adjacency matrix.
    check_connected : bool, default=True
        If True, raise an error when the graph is disconnected.
    unweighted : bool, default=True
        Treat the graph as unweighted.
    directed : bool, default=False
        Whether to treat the graph as directed.
    return_predecessors : bool, default=False
        Passed through to scipy.sparse.csgraph.shortest_path.

    Returns
    -------
    D : np.ndarray
        Dense distance matrix of shape (n, n).
    """
    A = sp.csr_matrix(A)

    if check_connected:
        n_components, _ = connected_components(A, directed=directed, connection="weak")
        if n_components != 1:
            raise ValueError(
                "Graph is disconnected, so D contains infinities. "
                "Use the largest connected component first."
            )

    result = shortest_path(
        A,
        directed=directed,
        unweighted=unweighted,
        return_predecessors=return_predecessors,
    )

    if return_predecessors:
        D, predecessors = result
        return D, predecessors

    D = result
    return D


def steinerberger_curvature_from_distance(
    D: np.ndarray,
    *,
    method: str = "solve",
    regularization: float | None = None,
    check_finite: bool = True,
    rhs_scale: float = 1.0,
) -> np.ndarray:
    """
    Solve the Steinerberger curvature system

        D x = rhs_scale * 1

    where D is the graph distance matrix and 1 is the all-ones vector.

    Parameters
    ----------
    D : np.ndarray
        Square distance matrix.
    method : str, default="solve"
        "solve" for direct symmetric solve, "lstsq" for least-squares.
    regularization : float | None, default=None
        Optional regularization parameter to add to diagonal of D.
    check_finite : bool, default=True
        If True, check that D contains no non-finite values.
    rhs_scale : float, default=1.0
        Scale factor for the right-hand side. Use rhs_scale=n to solve Dx=n*1.
    """
    D = np.asarray(D, dtype=np.float64)

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square matrix.")

    if check_finite and not np.all(np.isfinite(D)):
        raise ValueError("D contains non-finite values. Is the graph disconnected?")

    n = D.shape[0]
    b = rhs_scale * np.ones(n, dtype=np.float64)

    M = D.copy()
    if regularization is not None and regularization > 0:
        M = M + regularization * np.eye(n, dtype=np.float64)

    if method == "solve":
        x = solve(M, b, assume_a="sym")
    elif method == "lstsq":
        x, *_ = lstsq(M, b)
    else:
        raise ValueError("method must be one of {'solve', 'lstsq'}")

    return x


def steinerberger_curvature(
    A: sp.spmatrix,
    *,
    use_lcc: bool = True,
    solver: str = "solve",
    regularization: float | None = None,
    rhs_scale: float = 1.0,
    return_distance: bool = False,
    return_nodes: bool = False,
) -> np.ndarray | tuple:
    """
    Convenience wrapper:
      1. optionally extract largest connected component
      2. compute all-pairs distance matrix D
      3. solve D x = rhs_scale * 1

    Parameters
    ----------
    A : sp.spmatrix
        Sparse adjacency matrix.
    use_lcc : bool, default=True
        If True, restrict to the largest connected component.
    solver : str, default="solve"
        "solve" for direct symmetric solve, "lstsq" for least-squares.
    regularization : float | None, default=None
        Optional regularization parameter.
    rhs_scale : float, default=1.0
        Scale factor for the right-hand side. Use rhs_scale=n to solve Dx=n*1.
    return_distance : bool, default=False
        If True, return the distance matrix D as well.
    return_nodes : bool, default=False
        If True, return the node indices.
    """
    A = sp.csr_matrix(A)
    nodes = None

    if use_lcc:
        A, nodes = largest_connected_component(A)

    D = distance_matrix_from_adjacency(A, check_connected=True)
    x = steinerberger_curvature_from_distance(
        D,
        method=solver,
        regularization=regularization,
        rhs_scale=rhs_scale,
    )

    outputs = [x]
    if return_distance:
        outputs.append(D)
    if return_nodes:
        if nodes is None:
            nodes = np.arange(A.shape[0])
        outputs.append(nodes)

    return outputs[0] if len(outputs) == 1 else tuple(outputs)


def resistance_distance_matrix_from_adjacency(
    A: sp.spmatrix,
    *,
    check_connected: bool = True,
) -> np.ndarray:
    """
    Compute the effective-resistance matrix Omega of a connected graph.

    Following the formulation used in "Graph curvature via resistance distance",
    let L be the graph Laplacian and J the all-ones matrix. For a connected
    graph, Gamma = L + J / n is invertible and the resistance distance is

        Omega_ij = Gamma^{-1}_{ii} + Gamma^{-1}_{jj} - 2 Gamma^{-1}_{ij}.

    Parameters
    ----------
    A : sp.spmatrix
        Sparse adjacency matrix of an undirected graph.
    check_connected : bool, default=True
        If True, raise when the graph is disconnected.

    Returns
    -------
    Omega : np.ndarray
        Dense effective-resistance matrix of shape (n, n).
    """
    A = sp.csr_matrix(A)

    if check_connected:
        n_components, _ = connected_components(A, directed=False, connection="weak")
        if n_components != 1:
            raise ValueError(
                "Graph is disconnected, so the resistance distance matrix is "
                "not defined globally. Use the largest connected component first."
            )

    n = A.shape[0]
    degrees = np.asarray(A.sum(axis=1)).ravel()
    L = np.diag(degrees) - A.toarray()
    Gamma = L + np.ones((n, n), dtype=np.float64) / float(n)
    Gamma_inv = np.linalg.inv(Gamma)

    diag = np.diag(Gamma_inv)
    Omega = diag[:, None] + diag[None, :] - 2.0 * Gamma_inv
    Omega = 0.5 * (Omega + Omega.T)
    np.fill_diagonal(Omega, 0.0)
    return Omega


def resistance_curvature_from_resistance_distance(
    Omega: np.ndarray,
    *,
    method: str = "solve",
    regularization: float | None = None,
    check_finite: bool = True,
    rhs_scale: float = 1.0,
) -> np.ndarray:
    """
    Solve the resistance curvature system

        Omega x = rhs_scale * 1,

    where Omega is the graph effective-resistance matrix.
    """
    Omega = np.asarray(Omega, dtype=np.float64)

    if Omega.ndim != 2 or Omega.shape[0] != Omega.shape[1]:
        raise ValueError("Omega must be a square matrix.")

    if check_finite and not np.all(np.isfinite(Omega)):
        raise ValueError("Omega contains non-finite values.")

    n = Omega.shape[0]
    b = rhs_scale * np.ones(n, dtype=np.float64)

    M = Omega.copy()
    if regularization is not None and regularization > 0:
        M = M + regularization * np.eye(n, dtype=np.float64)

    if method == "solve":
        x = solve(M, b, assume_a="sym")
    elif method == "lstsq":
        x, *_ = lstsq(M, b)
    else:
        raise ValueError("method must be one of {'solve', 'lstsq'}")

    return x


def resistance_curvature(
    A: sp.spmatrix,
    *,
    use_lcc: bool = True,
    solver: str = "solve",
    regularization: float | None = None,
    rhs_scale: float = 1.0,
    return_resistance_distance: bool = False,
    return_nodes: bool = False,
) -> np.ndarray | tuple:
    """
    Convenience wrapper:
      1. optionally extract largest connected component
      2. compute the effective-resistance matrix Omega
      3. solve Omega x = rhs_scale * 1

    Parameters
    ----------
    A : sp.spmatrix
        Sparse adjacency matrix.
    use_lcc : bool, default=True
        If True, restrict to the largest connected component.
    solver : str, default="solve"
        "solve" for direct symmetric solve, "lstsq" for least-squares.
    regularization : float | None, default=None
        Optional regularization parameter.
    rhs_scale : float, default=1.0
        Scale factor for the right-hand side.
    return_resistance_distance : bool, default=False
        If True, return the resistance distance matrix Omega as well.
    return_nodes : bool, default=False
        If True, return the node indices.
    """
    A = sp.csr_matrix(A)
    nodes = None

    if use_lcc:
        A, nodes = largest_connected_component(A)

    Omega = resistance_distance_matrix_from_adjacency(A, check_connected=True)
    x = resistance_curvature_from_resistance_distance(
        Omega,
        method=solver,
        regularization=regularization,
        rhs_scale=rhs_scale,
    )

    outputs = [x]
    if return_resistance_distance:
        outputs.append(Omega)
    if return_nodes:
        if nodes is None:
            nodes = np.arange(A.shape[0])
        outputs.append(nodes)

    return outputs[0] if len(outputs) == 1 else tuple(outputs)


def _wasserstein_1_lp(mu: np.ndarray, nu: np.ndarray, cost: np.ndarray) -> float:
    """Compute Wasserstein-1 distance between two discrete measures by LP."""
    mu = np.asarray(mu, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    m, n = cost.shape
    if mu.shape != (m,) or nu.shape != (n,):
        raise ValueError("Incompatible measure and cost dimensions.")

    mu = mu / mu.sum()
    nu = nu / nu.sum()

    a_rows = np.zeros((m, m * n), dtype=np.float64)
    for i in range(m):
        a_rows[i, i * n:(i + 1) * n] = 1.0

    a_cols = np.zeros((n, m * n), dtype=np.float64)
    for j in range(n):
        a_cols[j, j::n] = 1.0

    a_eq = np.vstack([a_rows, a_cols])
    b_eq = np.concatenate([mu, nu])

    result = linprog(
        c=cost.ravel(),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
    )

    if not result.success:
        raise RuntimeError(f"Optimal transport LP failed: {result.message}")

    return float(result.fun)


def _sampled_ollivier_edge_curvatures(
    A: sp.spmatrix,
    *,
    alpha: float,
    max_neighbors: int,
    edge_sample_size: int | None,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate edge-wise Ollivier-Ricci curvature on a sampled edge set."""
    if not (0.0 <= alpha < 1.0):
        raise ValueError("alpha must satisfy 0 <= alpha < 1.")
    if max_neighbors <= 0:
        raise ValueError("max_neighbors must be positive.")

    A = sp.csr_matrix(A)
    rng = np.random.default_rng(seed)

    D = distance_matrix_from_adjacency(A, check_connected=True, unweighted=True)

    uu, vv = sp.triu(A, k=1).nonzero()
    edges = np.column_stack([uu, vv]).astype(int, copy=False)
    if edges.shape[0] == 0:
        raise ValueError("Graph has no edges.")

    if edge_sample_size is None or edge_sample_size >= edges.shape[0]:
        sampled_edges = edges
    else:
        idx = rng.choice(edges.shape[0], size=edge_sample_size, replace=False)
        sampled_edges = edges[idx]

    kappas = np.empty(sampled_edges.shape[0], dtype=np.float64)

    for i, (u, v) in enumerate(sampled_edges):
        nbrs_u = np.asarray(A.indices[A.indptr[u]:A.indptr[u + 1]], dtype=int)
        nbrs_v = np.asarray(A.indices[A.indptr[v]:A.indptr[v + 1]], dtype=int)

        if nbrs_u.size > max_neighbors:
            nbrs_u = rng.choice(nbrs_u, size=max_neighbors, replace=False)
        if nbrs_v.size > max_neighbors:
            nbrs_v = rng.choice(nbrs_v, size=max_neighbors, replace=False)

        supp_u = np.unique(np.concatenate([[u], nbrs_u]))
        supp_v = np.unique(np.concatenate([[v], nbrs_v]))

        mu = np.full(supp_u.size, (1.0 - alpha) / max(supp_u.size - 1, 1), dtype=np.float64)
        nu = np.full(supp_v.size, (1.0 - alpha) / max(supp_v.size - 1, 1), dtype=np.float64)

        mu[np.where(supp_u == u)[0][0]] = alpha
        nu[np.where(supp_v == v)[0][0]] = alpha

        if supp_u.size == 1:
            mu[0] = 1.0
        if supp_v.size == 1:
            nu[0] = 1.0

        C = D[np.ix_(supp_u, supp_v)]
        w1 = _wasserstein_1_lp(mu, nu, C)
        kappas[i] = 1.0 - w1

    return sampled_edges, kappas


def ollivier_ricci_curvature(
    A: sp.spmatrix,
    *,
    alpha: float = 0.5,
    max_neighbors: int = 16,
    edge_sample_size: int | None = 800,
    seed: int | None = None,
    return_edge_curvature: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximate Ollivier-Ricci node curvature for an unweighted graph.

    For sampled edges (u,v), this estimates
        kappa_alpha(u,v) = 1 - W1(m_u^alpha, m_v^alpha),
    where m_u^alpha is the lazy random-walk measure at u.
    """
    A = sp.csr_matrix(A)
    n = A.shape[0]

    sampled_edges, edge_kappa = _sampled_ollivier_edge_curvatures(
        A,
        alpha=alpha,
        max_neighbors=max_neighbors,
        edge_sample_size=edge_sample_size,
        seed=seed,
    )

    node_sum = np.zeros(n, dtype=np.float64)
    node_cnt = np.zeros(n, dtype=np.int64)
    for (u, v), kappa in zip(sampled_edges, edge_kappa):
        node_sum[u] += kappa
        node_sum[v] += kappa
        node_cnt[u] += 1
        node_cnt[v] += 1

    node_curv = np.empty(n, dtype=np.float64)
    mask = node_cnt > 0
    node_curv[mask] = node_sum[mask] / node_cnt[mask]
    fill = float(node_curv[mask].mean()) if np.any(mask) else 0.0
    node_curv[~mask] = fill

    if return_edge_curvature:
        return node_curv, sampled_edges, edge_kappa
    return node_curv


def lin_lu_yau_curvature(
    A: sp.spmatrix,
    *,
    alpha_near_one: float = 0.99,
    max_neighbors: int = 16,
    edge_sample_size: int | None = 800,
    seed: int | None = None,
    return_edge_curvature: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximate Lin-Lu-Yau node curvature for an unweighted graph.

    Uses the near-one idleness approximation
        kappa_LLY(u,v) ~= kappa_alpha(u,v) / (1 - alpha),
    with alpha close to 1.
    """
    if not (0.0 < alpha_near_one < 1.0):
        raise ValueError("alpha_near_one must satisfy 0 < alpha_near_one < 1.")

    A = sp.csr_matrix(A)
    n = A.shape[0]

    sampled_edges, edge_kappa_alpha = _sampled_ollivier_edge_curvatures(
        A,
        alpha=alpha_near_one,
        max_neighbors=max_neighbors,
        edge_sample_size=edge_sample_size,
        seed=seed,
    )
    edge_kappa_lly = edge_kappa_alpha / (1.0 - alpha_near_one)

    node_sum = np.zeros(n, dtype=np.float64)
    node_cnt = np.zeros(n, dtype=np.int64)
    for (u, v), kappa in zip(sampled_edges, edge_kappa_lly):
        node_sum[u] += kappa
        node_sum[v] += kappa
        node_cnt[u] += 1
        node_cnt[v] += 1

    node_curv = np.empty(n, dtype=np.float64)
    mask = node_cnt > 0
    node_curv[mask] = node_sum[mask] / node_cnt[mask]
    fill = float(node_curv[mask].mean()) if np.any(mask) else 0.0
    node_curv[~mask] = fill

    if return_edge_curvature:
        return node_curv, sampled_edges, edge_kappa_lly
    return node_curv
