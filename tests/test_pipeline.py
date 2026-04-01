from __future__ import annotations

import unittest

import numpy as np
import scipy.sparse as sp

from facialgen.data import CyclicFaceChunkDataset, FacialWalkVertexDataset
from facialgen.early_stopping import (
    connected_link_prediction_split,
    edge_overlap_ratio,
)
from facialgen.evaluation import (
    average_rank_from_graph_statistics,
    compute_graph_statistics,
    reconstruct_graph_from_generated_walks,
)


def toy_graph() -> tuple[sp.csr_matrix, np.ndarray]:
    A = sp.csr_matrix(
        np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [0, 1, 1, 0],
            ],
            dtype=float,
        )
    )
    curvature = np.array([0.1, -0.2, 0.3, 0.4], dtype=float)
    return A, curvature


class FacialWalkDatasetSmokeTests(unittest.TestCase):
    def test_full_face_sequences_are_boundary_free(self) -> None:
        A, curvature = toy_graph()
        face_ds = FacialWalkVertexDataset(
            A,
            curvature,
            num_sign_configs=3,
            sign_seed=7,
        )

        self.assertGreater(len(face_ds), 0)
        sample = face_ds[0]

        self.assertEqual(sample["sequence_length"], int(sample["tokens"].numel()))
        self.assertTrue(np.all(sample["tokens"].numpy() < face_ds.num_nodes))
        self.assertNotIn(face_ds.bos_token_id, sample["tokens"].tolist())
        self.assertIn(sample["sign_config_index"], range(3))
        self.assertGreaterEqual(sample["face_index_within_config"], 0)

    def test_chunk_dataset_changes_rotation_across_epochs(self) -> None:
        A, curvature = toy_graph()
        face_ds = FacialWalkVertexDataset(
            A,
            curvature,
            num_sign_configs=2,
            sign_seed=11,
        )
        chunk_ds = CyclicFaceChunkDataset(
            face_ds,
            vertex_context_size=5,
            dart_stride=1,
            epoch_seed=19,
        )

        face_idx = 0
        original_darts = face_ds.dart_faces[face_idx]
        self.assertGreater(len(original_darts), 2)

        def collect_chunks(epoch: int) -> list[dict[str, object]]:
            chunk_ds.set_epoch(epoch)
            pieces = []
            for idx, (mapped_face_idx, _, _) in enumerate(chunk_ds.chunk_to_face):
                if mapped_face_idx == face_idx:
                    item = chunk_ds[idx]
                    pieces.append(
                        {
                            "tokens": item["tokens"].numpy(),
                            "chunk_start": int(item["chunk_start"]),
                            "dart_length": int(item["dart_length"]),
                        }
                    )
            return pieces

        chunks_e0 = collect_chunks(0)
        chunks_e1 = collect_chunks(1)
        seq_e0 = chunks_e0[0]["tokens"]
        seq_e1 = chunks_e1[0]["tokens"]

        self.assertEqual(seq_e0[0], face_ds.bos_token_id)
        self.assertEqual(seq_e1[0], face_ds.bos_token_id)
        self.assertEqual(chunks_e0[0]["chunk_start"], 0)
        self.assertEqual(chunks_e1[0]["chunk_start"], 0)
        self.assertGreater(len(chunks_e0), 1)
        self.assertEqual(chunks_e0[1]["chunk_start"] - chunks_e0[0]["chunk_start"], 1)
        self.assertLessEqual(len(chunks_e0[0]["tokens"]), 5)
        self.assertEqual(chunks_e0[0]["dart_length"], 2)
        self.assertEqual(
            chunks_e0[-1]["chunk_start"] + chunks_e0[-1]["dart_length"],
            len(original_darts),
        )
        self.assertNotIn(face_ds.eos_token_id, chunks_e0[0]["tokens"].tolist())
        self.assertNotIn(face_ds.eos_token_id, chunks_e0[-1]["tokens"].tolist())
        covered_e0 = set()
        covered_e1 = set()
        for item in chunks_e0:
            start = item["chunk_start"]
            covered_e0.update(range(start, start + item["dart_length"]))
        for item in chunks_e1:
            start = item["chunk_start"]
            covered_e1.update(range(start, start + item["dart_length"]))
        self.assertEqual(covered_e0, set(range(len(original_darts))))
        self.assertEqual(covered_e1, set(range(len(original_darts))))
        self.assertFalse(np.array_equal(seq_e0, seq_e1))

    def test_chunk_tokens_match_rotated_dart_windows(self) -> None:
        A, curvature = toy_graph()
        face_ds = FacialWalkVertexDataset(
            A,
            curvature,
            num_sign_configs=1,
            sign_seed=5,
        )
        chunk_ds = CyclicFaceChunkDataset(
            face_ds,
            vertex_context_size=7,
            dart_stride=2,
            epoch_seed=13,
        )

        face_idx = 0
        chunk_ds.set_epoch(0)
        rotated_darts = chunk_ds._rotated_dart_face(face_idx)
        bos = face_ds.bos_token_id

        matching_indices = [
            idx
            for idx, (mapped_face_idx, _, _) in enumerate(chunk_ds.chunk_to_face)
            if mapped_face_idx == face_idx
        ]
        self.assertGreater(len(matching_indices), 0)

        for idx in matching_indices:
            item = chunk_ds[idx]
            start = int(item["chunk_start"])
            dart_length = int(item["dart_length"])
            expected = [bos]
            for u, v in rotated_darts[start:start + dart_length]:
                expected.extend([u, v])
            if bool(item["has_eos"]):
                expected.append(face_ds.eos_token_id)
            self.assertEqual(item["tokens"].tolist(), expected)
            self.assertLessEqual(len(expected), chunk_ds.vertex_context_size)

    def test_short_faces_can_emit_eos(self) -> None:
        A, curvature = toy_graph()
        face_ds = FacialWalkVertexDataset(
            A,
            curvature,
            num_sign_configs=1,
            sign_seed=17,
        )
        chunk_ds = CyclicFaceChunkDataset(
            face_ds,
            vertex_context_size=32,
            dart_stride=15,
            epoch_seed=23,
        )

        found_short_face = False
        for idx in range(len(chunk_ds)):
            item = chunk_ds[idx]
            face_idx = int(item["face_index"])
            if len(face_ds.dart_faces[face_idx]) <= chunk_ds.max_darts_per_chunk:
                found_short_face = True
                self.assertTrue(bool(item["has_eos"]))
                self.assertEqual(item["tokens"][-1].item(), face_ds.eos_token_id)
                break

        self.assertTrue(found_short_face)


class EvaluationSmokeTests(unittest.TestCase):
    def test_connected_link_prediction_split_keeps_train_connected(self) -> None:
        A = sp.csr_matrix(
            np.array(
                [
                    [0, 1, 1, 0, 0],
                    [1, 0, 1, 1, 0],
                    [1, 1, 0, 1, 1],
                    [0, 1, 1, 0, 1],
                    [0, 0, 1, 1, 0],
                ],
                dtype=float,
            )
        )
        split = connected_link_prediction_split(
            A,
            val_fraction=0.1,
            test_fraction=0.1,
            seed=3,
        )
        train_adj = split["train_adj"]
        n_components, _ = sp.csgraph.connected_components(train_adj, directed=False)
        self.assertEqual(n_components, 1)
        self.assertGreater(len(split["val_edges"]), 0)
        self.assertGreater(len(split["test_edges"]), 0)

    def test_reconstruct_graph_from_generated_walks(self) -> None:
        walks = [
            [4, 0, 1, 2, 3, 5],
            [4, 1, 2, 1, 0, 5],
            [4, 2, 3, 2, 1, 5],
        ]

        A_hat, S = reconstruct_graph_from_generated_walks(
            walks,
            num_nodes=4,
            target_num_edges=3,
            seed=0,
        )

        self.assertEqual(S.shape, (4, 4))
        self.assertEqual(A_hat.shape, (4, 4))
        self.assertEqual((A_hat != A_hat.T).nnz, 0)
        self.assertEqual(A_hat.diagonal().sum(), 0.0)
        self.assertTrue(np.all(np.isin(A_hat.data, [1.0])))
        self.assertGreaterEqual(A_hat.nnz, 2)

    def test_edge_overlap_ratio(self) -> None:
        A_ref = sp.csr_matrix(
            np.array(
                [
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 0],
                ],
                dtype=float,
            )
        )
        A_gen = sp.csr_matrix(
            np.array(
                [
                    [0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0],
                ],
                dtype=float,
            )
        )
        self.assertAlmostEqual(edge_overlap_ratio(A_gen, A_ref), 0.5)

    def test_graph_statistics_on_triangle_graph(self) -> None:
        A = sp.csr_matrix(
            np.array(
                [
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0],
                ],
                dtype=float,
            )
        )
        labels = np.array([0, 0, 1])
        stats = compute_graph_statistics(A, labels=labels)

        self.assertEqual(stats["max_degree"], 2.0)
        self.assertEqual(stats["triangle_count"], 1.0)
        self.assertAlmostEqual(stats["clustering_coeff"], 1.0)
        self.assertAlmostEqual(stats["characteristic_path_len"], 1.0)
        self.assertAlmostEqual(stats["intra_community_density"], 1.0)
        self.assertAlmostEqual(stats["inter_community_density"], 1.0)

    def test_average_rank_helper(self) -> None:
        ref = {"max_degree": 2.0, "assortativity": 0.0}
        cands = [
            {"max_degree": 2.0, "assortativity": 0.1},
            {"max_degree": 3.0, "assortativity": 0.0},
        ]
        avg_rank, per_metric = average_rank_from_graph_statistics(ref, cands)

        self.assertEqual(avg_rank.shape, (2,))
        self.assertEqual(set(per_metric.keys()), {"max_degree", "assortativity"})
        self.assertTrue(np.all(avg_rank >= 1.0))


if __name__ == "__main__":
    unittest.main()
