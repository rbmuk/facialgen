from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from .rotation_systems import (
    dart_face_to_vertex_sequence,
    facial_walks_from_curvature_signs,
)

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - torch is optional for this package
    torch = None
    DataLoader = Any  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        pass


def load_coraml_sparse(
    *,
    data_dir: str | Path = "data",
    pyg_root: str | Path | None = None,
) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """
    Load CoraML as sparse matrices using PyTorch Geometric only.

    Parameters
    ----------
    data_dir : str or Path, default="data"
        Base directory used for local dataset cache.
    pyg_root : str or Path or None, default=None
        Root directory for PyG dataset cache. If None, uses data_dir/pyg.

    Returns
    -------
    A : sp.csr_matrix
        Sparse adjacency matrix of shape (n, n).
    X : sp.csr_matrix
        Sparse node feature matrix of shape (n, d).
    y : np.ndarray
        Integer class labels of shape (n,).

    Raises
    ------
    ImportError
        If torch_geometric is not installed.
    """
    try:
        datasets_mod = importlib.import_module("torch_geometric.datasets")
        CitationFull = getattr(datasets_mod, "CitationFull")
    except ImportError as exc:
        raise ImportError(
            "load_coraml_sparse requires torch_geometric. Install PyG to use this loader."
        ) from exc

    root = Path(pyg_root) if pyg_root is not None else Path(data_dir) / "pyg"
    dataset = CitationFull(root=str(root), name="cora_ml")
    data = dataset[0]

    num_nodes = int(data.num_nodes)
    edge_index = data.edge_index

    rows = edge_index[0].cpu().numpy()
    cols = edge_index[1].cpu().numpy()
    vals = np.ones(rows.shape[0], dtype=np.float64)

    A = sp.coo_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    A.data[:] = 1.0
    A.eliminate_zeros()

    x = data.x
    if getattr(x, "is_sparse", False):
        x = x.to_dense()
    X = sp.csr_matrix(x.cpu().numpy())

    y = data.y.cpu().numpy()
    return A, X, y


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for facial-walk datasets and dataloaders."
        )


def _validate_context_size(context_size: int) -> int:
    context_size = int(context_size)
    if context_size <= 0:
        raise ValueError("context_size must be positive.")
    return context_size


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


def _rotate_cycle(vertices: np.ndarray, offset: int) -> np.ndarray:
    if vertices.size == 0:
        return vertices.copy()
    offset %= vertices.size
    if offset == 0:
        return vertices.copy()
    return np.concatenate((vertices[offset:], vertices[:offset]))


def _num_chunks(sequence_length: int, context_size: int) -> int:
    return max(1, math.ceil(sequence_length / context_size))


def build_face_vertex_sequences(
    A: sp.spmatrix,
    curvature: np.ndarray,
    *,
    num_sign_configs: int,
    sign_seed: int | None = None,
    signs: np.ndarray | None = None,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
) -> dict[str, Any]:
    """
    Generate full vertex-token facial walks from curvature-sign rotations.

    The returned sequences are the complete face vertex cycles with special
    boundary tokens added:

        [BOS] v_0 v_1 ... v_k [EOS]
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
    eos_token_id = num_nodes + 1 if eos_token_id is None else int(eos_token_id)

    sequences: list[np.ndarray] = []
    sign_config_index: list[int] = []
    face_index_within_config: list[int] = []

    for config_idx, sign_vec in enumerate(sign_matrix):
        dart_faces = facial_walks_from_curvature_signs(A, curvature, sign_vec)
        for face_idx, dart_face in enumerate(dart_faces):
            vertex_face = np.asarray(
                dart_face_to_vertex_sequence(dart_face),
                dtype=np.int64,
            )
            tokens = np.empty(vertex_face.size + 2, dtype=np.int64)
            tokens[0] = bos_token_id
            tokens[1:-1] = vertex_face
            tokens[-1] = eos_token_id

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

    Each item is one complete facial walk represented as

        [BOS] v_0 v_1 ... v_k [EOS]
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
        eos_token_id: int | None = None,
    ) -> None:
        _require_torch()

        built = build_face_vertex_sequences(
            A,
            curvature,
            num_sign_configs=num_sign_configs,
            sign_seed=sign_seed,
            signs=signs,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
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
    Context-limited chunks from full face sequences with epoch-wise cyclic shifts.

    At each epoch, the vertex portion of each face is rotated by a random cyclic
    offset. The rotated full sequence

        [BOS] v'_0 v'_1 ... v'_k [EOS]

    is then partitioned into contiguous chunks of length at most `context_size`.
    """

    def __init__(
        self,
        face_dataset: FacialWalkVertexDataset,
        *,
        context_size: int,
        epoch_seed: int = 0,
        pad_token_id: int | None = None,
    ) -> None:
        _require_torch()

        self.face_dataset = face_dataset
        self.context_size = _validate_context_size(context_size)
        self.epoch_seed = int(epoch_seed)
        self.epoch = 0
        self.pad_token_id = (
            face_dataset.vocab_size if pad_token_id is None else int(pad_token_id)
        )

        self.chunk_to_face: list[tuple[int, int]] = []
        self.num_chunks_per_face = np.empty(len(face_dataset), dtype=np.int64)

        for face_idx, seq in enumerate(face_dataset.sequences):
            n_chunks = _num_chunks(len(seq), self.context_size)
            self.num_chunks_per_face[face_idx] = n_chunks
            for chunk_idx in range(n_chunks):
                self.chunk_to_face.append((face_idx, chunk_idx))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.chunk_to_face)

    def _cyclic_offset(self, face_idx: int, num_vertices: int) -> int:
        if num_vertices <= 1:
            return 0

        seed = (
            self.epoch_seed
            + 1_000_003 * self.epoch
            + 97_003 * face_idx
        )
        rng = np.random.default_rng(seed)
        return int(rng.integers(0, num_vertices))

    def _rotated_full_sequence(self, face_idx: int) -> np.ndarray:
        full_seq = self.face_dataset.sequences[face_idx]
        bos = int(full_seq[0])
        eos = int(full_seq[-1])
        vertices = np.asarray(full_seq[1:-1], dtype=np.int64)
        offset = self._cyclic_offset(face_idx, vertices.size)
        rotated_vertices = _rotate_cycle(vertices, offset)

        rotated = np.empty_like(full_seq)
        rotated[0] = bos
        rotated[1:-1] = rotated_vertices
        rotated[-1] = eos
        return rotated

    def __getitem__(self, idx: int) -> dict[str, Any]:
        face_idx, chunk_idx = self.chunk_to_face[idx]
        rotated = self._rotated_full_sequence(face_idx)

        start = chunk_idx * self.context_size
        stop = min(start + self.context_size, len(rotated))
        chunk = rotated[start:stop]

        return {
            "tokens": torch.as_tensor(chunk, dtype=torch.long),
            "face_index": int(face_idx),
            "chunk_index": int(chunk_idx),
            "num_chunks_for_face": int(self.num_chunks_per_face[face_idx]),
            "sign_config_index": int(self.face_dataset.sign_config_index[face_idx]),
            "face_index_within_config": int(
                self.face_dataset.face_index_within_config[face_idx]
            ),
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
                [item["face_index"] for item in batch],
                dtype=torch.long,
            ),
            "chunk_index": torch.tensor(
                [item["chunk_index"] for item in batch],
                dtype=torch.long,
            ),
            "num_chunks_for_face": torch.tensor(
                [item["num_chunks_for_face"] for item in batch],
                dtype=torch.long,
            ),
            "sign_config_index": torch.tensor(
                [item["sign_config_index"] for item in batch],
                dtype=torch.long,
            ),
            "face_index_within_config": torch.tensor(
                [item["face_index_within_config"] for item in batch],
                dtype=torch.long,
            ),
        }


def make_face_chunk_dataloader(
    chunk_dataset: CyclicFaceChunkDataset,
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
