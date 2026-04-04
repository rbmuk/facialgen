from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
from tqdm.auto import tqdm


def _to_undirected_simple_csr(A: sp.spmatrix) -> sp.csr_matrix:
    A = sp.csr_matrix(A)
    A = A.maximum(A.T).astype(np.float64)
    A.setdiag(0.0)
    A.eliminate_zeros()
    if A.nnz > 0:
        A.data[:] = 1.0
    return A


def _extract_valid_vertex_tokens(
    walk: Sequence[int],
    num_nodes: int,
) -> list[int]:
    return [int(v) for v in walk if 0 <= int(v) < num_nodes]


def transition_count_matrix_from_walks(
    walks: Iterable[Sequence[int]],
    *,
    num_nodes: int,
    walk_type: str = "facial",
) -> sp.csr_matrix:
    """
    Count dart transitions across faithful generated walks.

    Tokens outside `[0, num_nodes)` are ignored, which lets the function work
    directly with sequences that may contain BOS/EOS/PAD tokens.

    In the faithful vertex encoding, only even-offset vertex pairs correspond
    to actual darts/edges:

        u0 u1 u2 u0 u3 u2 ...

    encodes darts

        (u0, u1), (u2, u0), (u3, u2), ...

    so pairs like `(u1, u2)` must NOT be counted as edges.
    """
    counts: dict[tuple[int, int], float] = {}

    walk_type = str(walk_type)

    for walk in walks:
        vertices = _extract_valid_vertex_tokens(walk, num_nodes)
        if walk_type == "facial":
            even_len = len(vertices) - (len(vertices) % 2)
            for idx in range(0, even_len, 2):
                u = int(vertices[idx])
                v = int(vertices[idx + 1])
                if u == v:
                    continue
                counts[(u, v)] = counts.get((u, v), 0.0) + 1.0
        elif walk_type == "random":
            for idx in range(0, max(len(vertices) - 1, 0)):
                u = int(vertices[idx])
                v = int(vertices[idx + 1])
                if u == v:
                    continue
                counts[(u, v)] = counts.get((u, v), 0.0) + 1.0
        else:
            raise ValueError(f"Unsupported walk_type={walk_type!r}")

    if not counts:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float64)

    rows = np.fromiter((k[0] for k in counts.keys()), dtype=np.int64)
    cols = np.fromiter((k[1] for k in counts.keys()), dtype=np.int64)
    vals = np.fromiter(counts.values(), dtype=np.float64)
    return sp.coo_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()


def symmetrize_transition_scores(S: sp.spmatrix) -> sp.csr_matrix:
    """Symmetrize a directed transition count matrix via directional summation."""
    S = sp.csr_matrix(S, dtype=np.float64)
    S = S + S.T
    S.setdiag(0.0)
    S.eliminate_zeros()
    return S


def aggregate_transition_scores(
    S: sp.spmatrix,
    *,
    mode: str = "sum",
) -> sp.csr_matrix:
    """Aggregate directional transition scores into undirected scores."""
    S = sp.csr_matrix(S, dtype=np.float64)
    mode = str(mode)
    if mode == "sum":
        S = S + S.T
    elif mode == "max":
        S = S.maximum(S.T)
    elif mode == "none":
        S = S.copy()
    else:
        raise ValueError(f"Unsupported score_symmetrization={mode!r}")
    S.setdiag(0.0)
    S.eliminate_zeros()
    return S


def sample_graph_from_scores(
    S: sp.spmatrix,
    *,
    target_num_edges: int | None = None,
    seed: int | None = None,
    walk_type: str = "facial",
    score_symmetrization: str | None = None,
    show_progress: bool = False,
    progress_desc: str = "graph reconstruction",
    debug: bool = False,
) -> sp.csr_matrix:
    """
    Convert transition scores to an undirected binary adjacency matrix.

    This follows the NetGAN-style post-processing described by the user:
    For random walks, we typically aggregate both directions into an undirected score.
    2. Try to give every node at least one edge by row-wise sampling.
    3. Sample the remaining edges without replacement from the global score mass.

    For facial walks, we keep the directed dart counts as-is and only collapse to
    undirected edges when adding sampled edges to the output graph.
    """
    walk_type = str(walk_type)
    if walk_type not in {"random", "facial"}:
        raise ValueError(f"Unsupported walk_type={walk_type!r}")
    if score_symmetrization is None:
        score_symmetrization = "sum" if walk_type == "random" else "none"
    if debug:
        print(
            f"[{progress_desc}] aggregating scores "
            f"(mode={score_symmetrization}, walk_type={walk_type})"
        )
    S = aggregate_transition_scores(S, mode=score_symmetrization)
    n = S.shape[0]
    rng = np.random.default_rng(seed)

    target_num_edges = int(target_num_edges) if target_num_edges is not None else None

    edges: set[tuple[int, int]] = set()

    # Step 1: ensure each node has at least one incident edge whenever possible
    # by row-wise sampling proportional to the symmetrized transition scores.
    row_iter = range(n)
    if show_progress:
        row_iter = tqdm(
            row_iter,
            total=n,
            desc=f"{progress_desc}: row seeding",
            leave=False,
        )
    for i in row_iter:
        row_start, row_stop = S.indptr[i], S.indptr[i + 1]
        nbrs = S.indices[row_start:row_stop]
        vals = S.data[row_start:row_stop]
        if nbrs.size == 0:
            continue

        candidate_edges: list[tuple[int, int]] = []
        candidate_weights: list[float] = []
        for j, weight in zip(nbrs.tolist(), vals.tolist()):
            j = int(j)
            if i == j:
                continue
            edge = (i, j) if i < j else (j, i)
            if edge in edges:
                continue
            candidate_edges.append(edge)
            candidate_weights.append(float(weight))

        if not candidate_edges:
            continue

        weights = np.asarray(candidate_weights, dtype=np.float64)
        total = float(weights.sum())
        if total <= 0.0:
            continue
        probs = weights / total
        idx = int(rng.choice(len(candidate_edges), p=probs))
        edges.add(candidate_edges[idx])
    if debug:
        print(f"[{progress_desc}] row seeding selected {len(edges)} edges")

    # Step 2: global edge sampling without replacement until target edge count.
    upper = sp.triu(S, k=1).tocoo()
    all_edges = list(zip(upper.row.tolist(), upper.col.tolist()))
    all_weights = upper.data.astype(np.float64, copy=True)

    if target_num_edges is None:
        target_num_edges = len(all_edges)

    target_num_edges = min(target_num_edges, len(all_edges))

    available_edges = [
        edge for edge in all_edges
        if edge not in edges
    ]
    if debug:
        print(
            f"[{progress_desc}] global fill from {len(available_edges)} candidate edges "
            f"toward target={target_num_edges}"
        )
    if available_edges and len(edges) < target_num_edges:
        weight_map = {edge: weight for edge, weight in zip(all_edges, all_weights)}
        remaining_target = min(
            max(target_num_edges - len(edges), 0),
            len(available_edges),
        )
        if remaining_target > 0:
            weights = np.asarray([weight_map[e] for e in available_edges], dtype=np.float64)
            total = float(weights.sum())
            if total > 0:
                probs = weights / total
                if show_progress:
                    fill_pbar = tqdm(
                        total=1,
                        desc=f"{progress_desc}: edge fill",
                        leave=True,
                    )
                print("Sampling")
                sampled_idx = rng.choice(
                    len(available_edges),
                    size=remaining_target,
                    replace=False,
                    p=probs,
                )
                print("Done sampling")
                sampled_idx = np.atleast_1d(sampled_idx).astype(np.int64, copy=False)
                print("Adding edges")
                for idx in sampled_idx.tolist():
                    edges.add(available_edges[int(idx)])
                if show_progress:
                    fill_pbar.update(1)
                    fill_pbar.close()
    if debug:
        print(f"[{progress_desc}] finished with {len(edges)} undirected edges")

    if not edges:
        return sp.csr_matrix((n, n), dtype=np.float64)

    rows = []
    cols = []
    for u, v in sorted(edges):
        rows.extend((u, v))
        cols.extend((v, u))

    data = np.ones(len(rows), dtype=np.float64)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A


def reconstruct_graph_from_transition_matrix(
    S: sp.spmatrix,
    *,
    target_num_edges: int | None = None,
    seed: int | None = None,
    walk_type: str = "facial",
    score_symmetrization: str | None = None,
    show_progress: bool = False,
    progress_desc: str = "graph reconstruction",
    debug: bool = False,
) -> sp.csr_matrix:
    """
    Build a synthetic undirected graph from a precomputed transition matrix.
    """
    return sample_graph_from_scores(
        S,
        target_num_edges=target_num_edges,
        seed=seed,
        walk_type=walk_type,
        score_symmetrization=score_symmetrization,
        show_progress=show_progress,
        progress_desc=progress_desc,
        debug=debug,
    )


def reconstruct_graph_from_generated_walks(
    walks: Iterable[Sequence[int]],
    *,
    num_nodes: int,
    target_num_edges: int | None = None,
    seed: int | None = None,
    walk_type: str = "facial",
    score_symmetrization: str | None = None,
    show_progress: bool = False,
    progress_desc: str = "graph reconstruction",
    debug: bool = False,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Build a synthetic undirected graph from generated walks.

    Returns `(A_hat, S)` where:
    - `S` is the directed transition count matrix
    - `A_hat` is the sampled undirected binary adjacency matrix
    """
    S = transition_count_matrix_from_walks(
        walks,
        num_nodes=num_nodes,
        walk_type=walk_type,
    )
    A_hat = reconstruct_graph_from_transition_matrix(
        S,
        target_num_edges=target_num_edges,
        seed=seed,
        walk_type=walk_type,
        score_symmetrization=score_symmetrization,
        show_progress=show_progress,
        progress_desc=progress_desc,
        debug=debug,
    )
    return A_hat, S


def max_degree(A: sp.spmatrix) -> int:
    A = _to_undirected_simple_csr(A)
    return int(np.asarray(A.sum(axis=1)).ravel().max(initial=0))


def degree_assortativity(A: sp.spmatrix) -> float:
    A = _to_undirected_simple_csr(A)
    upper = sp.triu(A, k=1).tocoo()
    if upper.nnz == 0:
        return 0.0

    deg = np.asarray(A.sum(axis=1)).ravel()
    du = deg[upper.row]
    dv = deg[upper.col]
    x = np.concatenate((du, dv))
    y = np.concatenate((dv, du))

    x_mean = float(x.mean())
    y_mean = float(y.mean())
    cov = float(np.mean((x - x_mean) * (y - y_mean)))
    x_std = float(x.std())
    y_std = float(y.std())
    if x_std == 0.0 or y_std == 0.0:
        return 0.0
    return cov / (x_std * y_std)


def triangle_count(A: sp.spmatrix) -> int:
    A = _to_undirected_simple_csr(A)
    if A.nnz == 0:
        return 0
    a3 = A @ A @ A
    return int(round(a3.diagonal().sum() / 6.0))


def clustering_coefficient(A: sp.spmatrix) -> float:
    A = _to_undirected_simple_csr(A)
    deg = np.asarray(A.sum(axis=1)).ravel()
    wedges = float(np.sum(deg * (deg - 1) / 2.0))
    if wedges == 0.0:
        return 0.0
    tri = float(triangle_count(A))
    return 3.0 * tri / wedges


def characteristic_path_length(A: sp.spmatrix) -> float:
    A = _to_undirected_simple_csr(A)
    n_components, labels = connected_components(A, directed=False)
    if A.shape[0] == 0:
        return 0.0

    if n_components > 1:
        counts = np.bincount(labels)
        comp_id = int(np.argmax(counts))
        nodes = np.flatnonzero(labels == comp_id)
        A = A[nodes][:, nodes]

    dist = shortest_path(A, directed=False, unweighted=True)
    finite = dist[np.isfinite(dist)]
    finite = finite[finite > 0]
    if finite.size == 0:
        return 0.0
    return float(finite.mean())


def power_law_exponent(A: sp.spmatrix, *, xmin: int = 1) -> float:
    """
    Estimate a degree power-law exponent.

    If the optional `powerlaw` package is installed, we use it. Otherwise,
    we fall back to a simple continuous MLE-style estimate on degrees >= xmin.
    """
    A = _to_undirected_simple_csr(A)
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg = deg[deg >= xmin]
    if deg.size == 0:
        return float("nan")

    try:
        import powerlaw  # type: ignore

        fit = powerlaw.Fit(deg, xmin=xmin, discrete=True, verbose=False)
        return float(fit.power_law.alpha)
    except ImportError:
        deg = deg.astype(np.float64)
        return 1.0 + deg.size / np.sum(np.log(deg / (xmin - 0.5)))


def intra_community_density(
    A: sp.spmatrix,
    labels: Sequence[int] | np.ndarray,
) -> float:
    A = _to_undirected_simple_csr(A)
    labels = np.asarray(labels)
    if labels.shape[0] != A.shape[0]:
        raise ValueError("labels length must equal number of nodes.")

    upper = sp.triu(A, k=1).tocoo()
    realized = 0
    possible = 0

    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        size = idx.size
        if size < 2:
            continue
        possible += size * (size - 1) // 2

    if possible == 0:
        return 0.0

    realized = int(np.sum(labels[upper.row] == labels[upper.col]))
    return float(realized / possible)


def inter_community_density(
    A: sp.spmatrix,
    labels: Sequence[int] | np.ndarray,
) -> float:
    A = _to_undirected_simple_csr(A)
    labels = np.asarray(labels)
    if labels.shape[0] != A.shape[0]:
        raise ValueError("labels length must equal number of nodes.")

    upper = sp.triu(A, k=1).tocoo()
    total_possible = A.shape[0] * (A.shape[0] - 1) // 2
    intra_possible = 0
    for label in np.unique(labels):
        size = int(np.sum(labels == label))
        intra_possible += size * (size - 1) // 2
    inter_possible = total_possible - intra_possible
    if inter_possible <= 0:
        return 0.0

    realized = int(np.sum(labels[upper.row] != labels[upper.col]))
    return float(realized / inter_possible)


def compute_graph_statistics(
    A: sp.spmatrix,
    *,
    labels: Sequence[int] | np.ndarray | None = None,
) -> dict[str, float]:
    stats = {
        "max_degree": float(max_degree(A)),
        "assortativity": float(degree_assortativity(A)),
        "triangle_count": float(triangle_count(A)),
        "power_law_exp": float(power_law_exponent(A)),
        "clustering_coeff": float(clustering_coefficient(A)),
        "characteristic_path_len": float(characteristic_path_length(A)),
    }

    if labels is not None:
        stats["inter_community_density"] = float(inter_community_density(A, labels))
        stats["intra_community_density"] = float(intra_community_density(A, labels))

    return stats


def average_rank_from_graph_statistics(
    reference_stats: dict[str, float],
    candidate_stats: Sequence[dict[str, float]],
    *,
    metric_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Rank candidates by per-metric absolute error to a reference graph.

    Returns:
    - average rank per candidate
    - per-metric rank vectors
    """
    if metric_names is None:
        metric_names = tuple(reference_stats.keys())

    num_candidates = len(candidate_stats)
    rank_matrix = {}

    for metric in metric_names:
        ref = float(reference_stats[metric])
        errors = np.empty(num_candidates, dtype=np.float64)
        for i, cand in enumerate(candidate_stats):
            val = float(cand[metric])
            if np.isnan(ref) or np.isnan(val):
                errors[i] = np.inf
            else:
                errors[i] = abs(val - ref)

        order = np.argsort(errors, kind="stable")
        ranks = np.empty(num_candidates, dtype=np.float64)
        ranks[order] = np.arange(1, num_candidates + 1, dtype=np.float64)
        rank_matrix[metric] = ranks

    stacked = np.vstack([rank_matrix[m] for m in metric_names])
    return stacked.mean(axis=0), rank_matrix
