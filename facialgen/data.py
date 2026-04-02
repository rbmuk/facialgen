from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from .rotation_systems import facial_walks_from_curvature_signs

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - torch is optional for this package
    torch = None
    DataLoader = Any  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        pass


def load_graph_dataset_sparse(
    dataset_name: str,
    *,
    data_dir: str | Path = "data",
    pyg_root: str | Path | None = None,
) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """
    Load one of the supported graph datasets as sparse matrices.

    Supported names:
    - `coraml`, `cora_ml`
    - `citeseer`
    - `polblogs`
    """
    dataset_key = dataset_name.strip().lower().replace("-", "_")

    try:
        datasets_mod = importlib.import_module("torch_geometric.datasets")
    except ImportError as exc:
        raise ImportError(
            "load_graph_dataset_sparse requires torch_geometric. Install PyG to use this loader."
        ) from exc

    root = Path(pyg_root) if pyg_root is not None else Path(data_dir) / "pyg"

    try:
        if dataset_key in {"coraml", "cora_ml"}:
            CitationFull = getattr(datasets_mod, "CitationFull")
            dataset = CitationFull(root=str(root), name="cora_ml")
        elif dataset_key == "citeseer":
            Planetoid = getattr(datasets_mod, "Planetoid")
            dataset = Planetoid(root=str(root), name="Citeseer")
        elif dataset_key == "polblogs":
            PolBlogs = getattr(datasets_mod, "PolBlogs")
            dataset = PolBlogs(root=str(root))
        else:
            raise ValueError(
                f"Unsupported dataset_name={dataset_name!r}. "
                "Supported values are: coraml, citeseer, polblogs."
            )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load dataset {dataset_name!r}. If it is not already cached "
            f"locally under {root}, PyG may be trying to download it."
        ) from exc

    data = dataset[0]

    num_nodes = int(data.num_nodes)
    edge_index = data.edge_index

    rows = edge_index[0].cpu().numpy()
    cols = edge_index[1].cpu().numpy()
    vals = np.ones(rows.shape[0], dtype=np.float64)

    A = sp.coo_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    A.data[:] = 1.0
    A.eliminate_zeros()

    x = getattr(data, "x", None)
    if x is None:
        X = sp.eye(num_nodes, format="csr", dtype=np.float64)
    else:
        if getattr(x, "is_sparse", False):
            x = x.to_dense()
        X = sp.csr_matrix(x.cpu().numpy())

    y = getattr(data, "y", None)
    if y is None:
        y_arr = np.full(num_nodes, -1, dtype=np.int64)
    else:
        y_arr = y.cpu().numpy()

    return A, X, y_arr


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for facial-walk datasets and dataloaders."
        )


def _validate_vertex_context_size(vertex_context_size: int) -> int:
    vertex_context_size = int(vertex_context_size)
    if vertex_context_size <= 0:
        raise ValueError("vertex_context_size must be positive.")
    return vertex_context_size


def _validate_dart_stride(dart_stride: int) -> int:
    dart_stride = int(dart_stride)
    if dart_stride <= 0:
        raise ValueError("dart_stride must be positive.")
    return dart_stride


def _sample_sign_configurations(
    num_sign_configs: int,
    num_nodes: int,
    *,
    seed: int | None = None,
) -> np.ndarray:
    if num_sign_configs <= 0:
        raise ValueError("num_sign_configs must be positive.")

    rng = np.random.default_rng(seed)
    return rng.choice(
        np.array([-1, 1], dtype=np.int8),
        size=(num_sign_configs, num_nodes),
        replace=True,
    )


def _dart_face_to_faithful_vertex_sequence(
    face: list[tuple[int, int]],
) -> np.ndarray:
    if not face:
        return np.empty(0, dtype=np.int64)

    seq = np.empty(2 * len(face), dtype=np.int64)
    for i, (u, v) in enumerate(face):
        seq[2 * i] = int(u)
        seq[2 * i + 1] = int(v)
    return seq


def _window_starts(
    sequence_length: int,
    window_length: int,
    dart_stride: int,
    *,
    allow_tail_overlap: bool = True,
) -> list[int]:
    if sequence_length <= window_length:
        return [0]

    starts = list(range(0, sequence_length - window_length + 1, dart_stride))
    if not starts:
        starts = [0]
    if not allow_tail_overlap:
        return starts

    tail_start = sequence_length - window_length
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


def _slice_cyclic_vertex_window(
    sequence: np.ndarray,
    *,
    start_token: int,
    token_length: int,
) -> np.ndarray:
    if token_length <= 0:
        return np.empty(0, dtype=sequence.dtype)
    seq_len = int(sequence.size)
    if seq_len == 0:
        return np.empty(0, dtype=sequence.dtype)
    start_token %= seq_len
    stop_token = start_token + token_length
    if stop_token <= seq_len:
        return sequence[start_token:stop_token]
    wrap = stop_token - seq_len
    return np.concatenate((sequence[start_token:], sequence[:wrap]))


def _faithful_vertex_sequence_to_dart_face(
    sequence: np.ndarray,
) -> list[tuple[int, int]]:
    if sequence.size == 0:
        return []
    if sequence.size % 2 != 0:
        raise ValueError("Faithful vertex sequence length must be even.")
    return [
        (int(sequence[2 * i]), int(sequence[2 * i + 1]))
        for i in range(sequence.size // 2)
    ]


def _csr_neighbor_lists(A: sp.spmatrix) -> list[np.ndarray]:
    A = sp.csr_matrix(A)
    return [
        A.indices[A.indptr[i]:A.indptr[i + 1]].astype(np.int64, copy=True)
        for i in range(A.shape[0])
    ]


def build_face_vertex_sequences(
    A: sp.spmatrix,
    curvature: np.ndarray,
    *,
    num_sign_configs: int,
    sign_seed: int | None = None,
    signs: np.ndarray | None = None,
    bos_token_id: int | None = None,
) -> dict[str, Any]:
    """
    Generate full facial walks from curvature-sign rotations.

    If a facial walk in dart form is

        (u_0, u_1) -> (u_2, u_0) -> (u_3, u_2) -> ...,

    then we store its faithful vertex-token encoding as

        u_0 u_1 u_2 u_0 u_3 u_2 ...
    """
    A = sp.csr_matrix(A)
    num_nodes = int(A.shape[0])

    curvature = np.asarray(curvature, dtype=float)
    if curvature.shape != (num_nodes,):
        raise ValueError(
            f"curvature must have shape ({num_nodes},), got {curvature.shape}"
        )

    if signs is None:
        sign_matrix = _sample_sign_configurations(
            num_sign_configs,
            num_nodes,
            seed=sign_seed,
        )
    else:
        sign_matrix = np.asarray(signs)
        if sign_matrix.ndim == 1:
            sign_matrix = sign_matrix[None, :]
        if sign_matrix.shape[1] != num_nodes:
            raise ValueError(
                f"signs must have shape (m, {num_nodes}), got {sign_matrix.shape}"
            )
        num_sign_configs = int(sign_matrix.shape[0])

    bos_token_id = num_nodes if bos_token_id is None else int(bos_token_id)
    eos_token_id = num_nodes + 1

    sequences: list[np.ndarray] = []
    sign_config_index: list[int] = []
    face_index_within_config: list[int] = []

    for config_idx, sign_vec in enumerate(sign_matrix):
        config_dart_faces = facial_walks_from_curvature_signs(A, curvature, sign_vec)
        for face_idx, dart_face in enumerate(config_dart_faces):
            vertex_face = _dart_face_to_faithful_vertex_sequence(list(dart_face))
            tokens = vertex_face.copy()

            sequences.append(tokens)
            sign_config_index.append(config_idx)
            face_index_within_config.append(face_idx)

    return {
        "sequences": sequences,
        "signs": np.asarray(sign_matrix, dtype=np.int8),
        "sign_config_index": np.asarray(sign_config_index, dtype=np.int64),
        "face_index_within_config": np.asarray(face_index_within_config, dtype=np.int64),
        "num_nodes": num_nodes,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
    }


class FacialWalkVertexDataset(Dataset):
    """
    Dataset of full face vertex sequences built from random +/- sign assignments.

    If the underlying facial walk is

        (u_0, u_1) -> (u_2, u_0) -> (u_3, u_2) -> ...,

    then each stored item is the faithful vertex-token sequence

        u_0 u_1 u_2 u_0 u_3 u_2 ...
    """

    def __init__(
        self,
        A: sp.spmatrix,
        curvature: np.ndarray,
        *,
        num_sign_configs: int,
        sign_seed: int | None = None,
        signs: np.ndarray | None = None,
        bos_token_id: int | None = None,
    ) -> None:
        _require_torch()

        built = build_face_vertex_sequences(
            A,
            curvature,
            num_sign_configs=num_sign_configs,
            sign_seed=sign_seed,
            signs=signs,
            bos_token_id=bos_token_id,
        )

        self.sequences: list[np.ndarray] = built["sequences"]
        self.signs: np.ndarray = built["signs"]
        self.sign_config_index: np.ndarray = built["sign_config_index"]
        self.face_index_within_config: np.ndarray = built["face_index_within_config"]
        self.num_nodes: int = built["num_nodes"]
        self.bos_token_id: int = built["bos_token_id"]
        self.eos_token_id: int = built["eos_token_id"]
        self.vocab_size: int = self.num_nodes + 2

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        tokens = torch.as_tensor(self.sequences[idx], dtype=torch.long)
        return {
            "tokens": tokens,
            "sign_config_index": int(self.sign_config_index[idx]),
            "face_index_within_config": int(self.face_index_within_config[idx]),
            "sequence_length": int(tokens.numel()),
        }


class CyclicFaceChunkDataset(Dataset):
    """
    Context-limited chunks from faithful full-face vertex sequences.

    The stored face representation is the faithful dart-endpoint vertex
    sequence, so each dart contributes two vertex tokens. At each epoch, each
    face is rotated by a random cyclic offset measured in darts. Chunk
    boundaries remain in dart units, but retrieval slices the already-stored
    faithful vertex sequence at even token offsets so `__getitem__` does only
    minimal wraparound slicing plus `BOS`/optional `EOS` handling. We append
    `EOS` only when the entire rotated face fits into a single sample, i.e. when
    the face is short enough that no chunking is needed. Long faces therefore
    contribute BOS-anchored interior or terminal fragments without `EOS`. This
    is valid because a facial walk is cyclic under dart rotation; after rotating
    the face, any chosen dart-window can be treated as a legitimate start of the
    sampled training view. In this dataset, `BOS` therefore marks the beginning
    of the sampled window, not a unique canonical start of the underlying face.
    Every training sample thus has the form

        [BOS] u_0 u_1 u_2 u_0 u_3 u_2 ... [EOS?]

    where the vertices come from one contiguous dart-window in the rotated
    faithful vertex sequence.

    For non-overlapping coverage in dart space, use

        dart_stride = (vertex_context_size - 1) // 2

    when `vertex_context_size` is odd, because ordinary chunked samples contain
    one `BOS` token plus two vertex tokens per dart. In this non-overlapping
    regime, the dataset does not
    force an extra overlapping tail chunk; a few darts at the end of a very long
    face may be omitted in a given epoch, but cyclic rotation changes which darts
    are omitted across epochs.
    """

    def __init__(
        self,
        face_dataset: FacialWalkVertexDataset,
        *,
        vertex_context_size: int,
        dart_stride: int | None = None,
        epoch_seed: int = 0,
        pad_token_id: int | None = None,
    ) -> None:
        _require_torch()

        self.face_dataset = face_dataset
        self.vertex_context_size = _validate_vertex_context_size(vertex_context_size)
        self.dart_stride = _validate_dart_stride(
            self.vertex_context_size // 2 if dart_stride is None else dart_stride
        )
        self.context_size = self.vertex_context_size
        self.stride = self.dart_stride
        self.epoch_seed = int(epoch_seed)
        self.epoch = 0
        self.pad_token_id = (
            face_dataset.vocab_size if pad_token_id is None else int(pad_token_id)
        )
        # Each ordinary chunk contains one BOS token plus two vertex tokens per
        # dart. EOS is reserved only for genuinely unchunked short faces.
        self.max_darts_per_chunk = max((self.vertex_context_size - 1) // 2, 1)
        self.allow_tail_overlap = self.dart_stride != self.max_darts_per_chunk
        self.face_lengths_vertices = np.asarray(
            [len(sequence) for sequence in face_dataset.sequences],
            dtype=np.int64,
        )
        self.face_lengths_darts = self.face_lengths_vertices // 2
        self.num_chunks_per_face = np.empty(len(face_dataset), dtype=np.int64)

        chunk_face_index: list[int] = []
        chunk_index: list[int] = []
        chunk_start: list[int] = []

        for face_idx, face_length_darts in enumerate(self.face_lengths_darts.tolist()):
            starts = _window_starts(
                face_length_darts,
                self.max_darts_per_chunk,
                self.dart_stride,
                allow_tail_overlap=self.allow_tail_overlap,
            )
            self.num_chunks_per_face[face_idx] = len(starts)
            for local_chunk_idx, start in enumerate(starts):
                chunk_face_index.append(face_idx)
                chunk_index.append(local_chunk_idx)
                chunk_start.append(start)

        self.chunk_face_index = np.asarray(chunk_face_index, dtype=np.int64)
        self.chunk_index = np.asarray(chunk_index, dtype=np.int64)
        self.chunk_start = np.asarray(chunk_start, dtype=np.int64)
        self.chunk_to_face: list[tuple[int, int, int]] = list(
            zip(
                self.chunk_face_index.tolist(),
                self.chunk_index.tolist(),
                self.chunk_start.tolist(),
            )
        )
        self.epoch_offsets_darts = np.zeros(len(face_dataset), dtype=np.int64)
        self.epoch_offsets_vertices = np.zeros(len(face_dataset), dtype=np.int64)
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        offsets_darts = np.zeros(len(self.face_dataset), dtype=np.int64)
        for face_idx, face_length in enumerate(self.face_lengths_darts.tolist()):
            offsets_darts[face_idx] = self._cyclic_offset(face_idx, face_length)
        self.epoch_offsets_darts = offsets_darts
        self.epoch_offsets_vertices = 2 * offsets_darts

    def __len__(self) -> int:
        return int(self.chunk_face_index.size)

    def _cyclic_offset(self, face_idx: int, face_length: int) -> int:
        if face_length <= 1:
            return 0

        seed = (
            self.epoch_seed
            + 1_000_003 * self.epoch
            + 97_003 * face_idx
        )
        rng = np.random.default_rng(seed)
        return int(rng.integers(0, face_length))

    def _rotated_dart_face(self, face_idx: int) -> list[tuple[int, int]]:
        rotated_vertices = _slice_cyclic_vertex_window(
            self.face_dataset.sequences[face_idx],
            start_token=int(self.epoch_offsets_vertices[face_idx]),
            token_length=int(self.face_lengths_vertices[face_idx]),
        )
        return _faithful_vertex_sequence_to_dart_face(rotated_vertices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        face_idx = int(self.chunk_face_index[idx])
        chunk_idx = int(self.chunk_index[idx])
        start = int(self.chunk_start[idx])
        face_length_darts = int(self.face_lengths_darts[face_idx])
        stop = min(start + self.max_darts_per_chunk, face_length_darts)
        token_length = 2 * (stop - start)
        token_start = int(self.epoch_offsets_vertices[face_idx]) + 2 * start
        vertex_chunk = _slice_cyclic_vertex_window(
            self.face_dataset.sequences[face_idx],
            start_token=token_start,
            token_length=token_length,
        )
        fits_without_chunking = face_length_darts <= self.max_darts_per_chunk
        add_eos = fits_without_chunking and stop == face_length_darts
        chunk = np.empty(vertex_chunk.size + 1 + int(add_eos), dtype=np.int64)
        chunk[0] = self.face_dataset.bos_token_id
        chunk[1:1 + vertex_chunk.size] = vertex_chunk
        if add_eos:
            chunk[-1] = self.face_dataset.eos_token_id

        return {
            "tokens": torch.as_tensor(chunk, dtype=torch.long),
            "face_index": int(face_idx),
            "chunk_index": int(chunk_idx),
            "chunk_start": int(start),
            "dart_length": int(stop - start),
            "is_terminal": bool(stop == face_length_darts),
            "has_eos": bool(add_eos),
            "num_chunks_for_face": int(self.num_chunks_per_face[face_idx]),
            "sign_config_index": int(self.face_dataset.sign_config_index[face_idx]),
            "face_index_within_config": int(
                self.face_dataset.face_index_within_config[face_idx]
            ),
            "dart_stride": int(self.dart_stride),
            "stride": int(self.dart_stride),
        }


class RandomWalkChunkDataset(Dataset):
    """
    Fixed-length BOS-anchored random walks sampled from a graph.

    Each sample has the form

        [BOS] v_0 v_1 ... v_{L-1}

    where consecutive vertex pairs are actual random-walk transitions in the
    graph. The walk is re-sampled each epoch from a deterministic epoch/index
    seed so repeated epochs expose different random walks without storing a huge
    precomputed corpus.
    """

    def __init__(
        self,
        A: sp.spmatrix,
        *,
        num_walks: int,
        vertex_context_size: int,
        epoch_seed: int = 0,
        bos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> None:
        _require_torch()

        self.A = sp.csr_matrix(A)
        self.num_nodes = int(self.A.shape[0])
        self.num_walks = int(num_walks)
        if self.num_walks <= 0:
            raise ValueError("num_walks must be positive.")
        self.vertex_context_size = _validate_vertex_context_size(vertex_context_size)
        self.walk_length = max(self.vertex_context_size - 1, 1)
        self.epoch_seed = int(epoch_seed)
        self.epoch = 0
        self.bos_token_id = self.num_nodes if bos_token_id is None else int(bos_token_id)
        self.eos_token_id = self.num_nodes + 1
        self.vocab_size = self.num_nodes + 2
        self.pad_token_id = self.vocab_size if pad_token_id is None else int(pad_token_id)
        self.neighbors = _csr_neighbor_lists(self.A)
        if any(len(nbrs) == 0 for nbrs in self.neighbors):
            raise ValueError("Random-walk dataset requires a graph with no isolated nodes.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_walks

    def _sample_walk_vertices(self, idx: int) -> np.ndarray:
        seed = self.epoch_seed + 1_000_003 * self.epoch + 97_003 * int(idx)
        rng = np.random.default_rng(seed)
        verts = np.empty(self.walk_length, dtype=np.int64)
        current = int(rng.integers(0, self.num_nodes))
        verts[0] = current
        for step in range(1, self.walk_length):
            nbrs = self.neighbors[current]
            current = int(nbrs[rng.integers(0, len(nbrs))])
            verts[step] = current
        return verts

    def __getitem__(self, idx: int) -> dict[str, Any]:
        vertices = self._sample_walk_vertices(int(idx))
        tokens = np.empty(vertices.size + 1, dtype=np.int64)
        tokens[0] = self.bos_token_id
        tokens[1:] = vertices
        return {
            "tokens": torch.as_tensor(tokens, dtype=torch.long),
            "face_index": int(idx),
            "chunk_index": 0,
            "chunk_start": 0,
            "dart_length": int(max(vertices.size - 1, 0)),
            "is_terminal": True,
            "has_eos": False,
            "num_chunks_for_face": 1,
            "sign_config_index": -1,
            "face_index_within_config": int(idx),
        }


class FaceChunkCollator:
    """Pad chunked face sequences for transformer-style training."""

    def __init__(self, pad_token_id: int) -> None:
        _require_torch()
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not batch:
            raise ValueError("Cannot collate an empty batch.")

        lengths = torch.tensor(
            [item["tokens"].numel() for item in batch],
            dtype=torch.long,
        )
        max_len = int(lengths.max().item())

        input_ids = torch.full(
            (len(batch), max_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

        for row_idx, item in enumerate(batch):
            seq = item["tokens"]
            seq_len = int(seq.numel())
            input_ids[row_idx, :seq_len] = seq
            attention_mask[row_idx, :seq_len] = True

        labels = input_ids.clone()
        labels[~attention_mask] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "lengths": lengths,
            "face_index": torch.tensor(
                [item.get("face_index", -1) for item in batch],
                dtype=torch.long,
            ),
            "chunk_index": torch.tensor(
                [item.get("chunk_index", -1) for item in batch],
                dtype=torch.long,
            ),
            "chunk_start": torch.tensor(
                [item.get("chunk_start", -1) for item in batch],
                dtype=torch.long,
            ),
            "dart_length": torch.tensor(
                [item.get("dart_length", -1) for item in batch],
                dtype=torch.long,
            ),
            "is_terminal": torch.tensor(
                [item.get("is_terminal", False) for item in batch],
                dtype=torch.bool,
            ),
            "has_eos": torch.tensor(
                [item.get("has_eos", False) for item in batch],
                dtype=torch.bool,
            ),
            "num_chunks_for_face": torch.tensor(
                [item.get("num_chunks_for_face", 1) for item in batch],
                dtype=torch.long,
            ),
            "sign_config_index": torch.tensor(
                [item.get("sign_config_index", -1) for item in batch],
                dtype=torch.long,
            ),
            "face_index_within_config": torch.tensor(
                [item.get("face_index_within_config", -1) for item in batch],
                dtype=torch.long,
            ),
        }


def make_face_chunk_dataloader(
    chunk_dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """Build a DataLoader for cyclically augmented facial-walk chunks."""
    _require_torch()

    return DataLoader(
        chunk_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=FaceChunkCollator(chunk_dataset.pad_token_id),
    )
