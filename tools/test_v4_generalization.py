"""CPU-only split-contract tests; no model training or performance claims."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v4.generalization import FAMILIES, GeneralizationPlan, field_metrics


class GeneralizationTests(unittest.TestCase):
    def spatial(self, **changes):
        options = dict(domain=((0, 1), (0, 1)), spatial_axes=(0, 1), mode="interpolation")
        return GeneralizationPlan(**(options | changes))

    def temporal(self, **changes):
        options = dict(
            domain=((-1, 1), (2, 12)), spatial_axes=(0,), time_axis=1,
            mode="time_extrapolation",
        )
        return GeneralizationPlan(**(options | changes))

    def test_sixteen_blocks(self):
        points = self.spatial().fixed_evaluation_points()
        self.assertEqual(len(points), 64)
        grid = GeneralizationPlan(((0, 1), (0, 1)), (0, 1)).fixed_evaluation_points()
        counts = np.bincount(self.spatial().spatial_block_ids(grid), minlength=16)
        np.testing.assert_array_equal(counts, np.full(16, 64))

    def test_sixteen_consecutive_seeds_visit_all_blocks(self):
        self.assertEqual({self.spatial(seed=s).block_id for s in range(3234, 3250)}, set(range(16)))

    def test_same_seed_same_region_across_methods(self):
        for seed in range(3234, 3238):
            plans = [self.spatial(seed=seed) for _ in range(6)]
            self.assertEqual(len({p.fingerprint() for p in plans}), 1)
            for plan in plans[1:]:
                np.testing.assert_array_equal(plan.fixed_evaluation_points(), plans[0].fixed_evaluation_points())

    def test_fixed_block_override(self):
        plans = [self.spatial(seed=s, block_override=6) for s in range(3234, 3238)]
        for plan in plans:
            self.assertEqual(plan.block_id, 6)
            np.testing.assert_array_equal(plan.fixed_evaluation_points(), plans[0].fixed_evaluation_points())

    def test_boundary_assignment(self):
        ids = self.spatial().spatial_block_ids([[0, 0], [1, 1], [0.25, 0.25], [0.5, 0]])
        np.testing.assert_array_equal(ids, [0, 15, 5, 2])

    def test_one_dimensional_space(self):
        plan = GeneralizationPlan(((-1, 1), (0, 1)), (0,), 1, "interpolation")
        selected = plan.fixed_evaluation_points(per_spatial_axis=32, time_count=5)
        self.assertEqual(len(selected), 10)
        self.assertTrue(np.all(plan.spatial_block_ids(selected) == plan.block_id))

    def test_coordinate_axis_order(self):
        plan = GeneralizationPlan(((0, 1), (-2, 2), (0, 1)), (1, 2), 0, "interpolation")
        points = plan.fixed_evaluation_points(time_count=3)
        self.assertEqual(len(points), 192)
        self.assertTrue(plan.evaluation_mask(points).all())

    def test_observation_labels_and_source_preserved(self):
        grid = GeneralizationPlan(((0, 1), (0, 1)), (0, 1)).fixed_evaluation_points()
        labels = np.column_stack((np.arange(len(grid)), np.arange(len(grid)) * 2))
        original = grid.copy(); original_labels = labels.copy()
        split = self.spatial().split_observations(grid, labels)
        self.assertEqual(len(split["heldout_coordinates"]), 64)
        self.assertEqual(len(split["train_coordinates"]), 960)
        np.testing.assert_array_equal(split["train_labels"], labels[split["train_mask"]])
        split["train_coordinates"][:] = 0
        np.testing.assert_array_equal(grid, original)
        np.testing.assert_array_equal(labels, original_labels)

    def test_interpolation_pde_bc_ic_preserved(self):
        plan = self.spatial(); points = plan.fixed_evaluation_points()
        self.assertFalse(plan.training_mask(points).any())
        for family in FAMILIES - {"observations"}:
            self.assertTrue(plan.training_mask(points, family).all())
            plan.assert_training(points, family)

    def test_interpolation_renewal_leak_rejected(self):
        plan = self.spatial()
        for _cycle in range(5):
            with self.assertRaises(ValueError):
                plan.assert_training(plan.fixed_evaluation_points())

    def test_sparse_or_empty_observation_split_rejected(self):
        for points, labels in [([], []), ([[0, 0]], [1]),
                               (self.spatial().fixed_evaluation_points(), np.ones(64))]:
            with self.assertRaises(ValueError):
                self.spatial().split_observations(points, labels)

    def test_temporal_cutoff_and_all_families(self):
        plan = self.temporal()
        self.assertEqual(plan.cutoff, 10)
        self.assertEqual(plan.training_domain, [[-1, 1], [2, 10]])
        points = [[0, 2], [0, 10], [0, np.nextafter(10, 12)], [0, 12]]
        for family in FAMILIES:
            np.testing.assert_array_equal(plan.training_mask(points, family), [True, True, False, False])
            with self.assertRaises(ValueError):
                plan.assert_training(points, family)
        np.testing.assert_array_equal(plan.evaluation_mask(points), [False, False, True, True])

    def test_temporal_grid_excludes_cutoff(self):
        plan = self.temporal(); points = plan.fixed_evaluation_points(time_count=4)
        self.assertTrue(np.all(points[:, 1] > 10))
        self.assertTrue(np.all(points[:, 1] < 12))
        self.assertEqual(len(points), 128)

    def test_stopping_validation_never_uses_heldout(self):
        for plan in (self.spatial(), self.temporal()):
            points = plan.fixed_evaluation_points()
            self.assertFalse(plan.validation_mask(points).any())
            self.assertFalse(plan.summary()["heldout_used_for_stopping"])

    def test_basic_does_not_remove_observations(self):
        plan = self.spatial(mode="basic")
        points = [[0, 0], [0.5, 0.5], [1, 1]]
        self.assertTrue(plan.training_mask(points).all())
        split = plan.split_observations(points, [1, 2, 3])
        self.assertEqual(len(split["heldout_labels"]), 0)

    def test_serialization_and_fingerprint(self):
        for plan in (self.spatial(), self.temporal()):
            restored = GeneralizationPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
            self.assertEqual(restored, plan)
            self.assertEqual(restored.fingerprint(), plan.fingerprint())
        self.assertNotEqual(self.spatial(seed=3234).fingerprint(), self.spatial(seed=3235).fingerprint())

    def test_grid_does_not_advance_global_rng(self):
        before = np.random.get_state()
        self.spatial().fixed_evaluation_points()
        after = np.random.get_state()
        self.assertEqual(before[0], after[0]); self.assertEqual(before[2:], after[2:])
        np.testing.assert_array_equal(before[1], after[1])

    def test_invalid_configuration(self):
        bad = [dict(seed=True), dict(seed=-1), dict(mode="combined"),
               dict(block_override=16), dict(block_override=True),
               dict(domain=((0, 0), (0, 1))), dict(spatial_axes=(0, 0)),
               dict(time_axis=0), dict(time_train_fraction=1), dict(version="unknown"),
               dict(mode="time_extrapolation")]
        for changes in bad:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.spatial(**changes)

    def test_invalid_points(self):
        for points in ([[1.1, 0]], [[float("nan"), 0]], [[0]], [0, 0]):
            with self.assertRaises(ValueError):
                self.spatial().training_mask(points)
        with self.assertRaises(ValueError):
            self.spatial().training_mask([[0, 0]], "unknown")

    def test_grid_resolution_guards(self):
        for count in (0, 5, 33, True, 2048):
            with self.assertRaises(ValueError):
                self.spatial().fixed_evaluation_points(per_spatial_axis=count)

    def test_field_metrics(self):
        report = field_metrics([1, 2], [1.1, 2.2])
        self.assertAlmostEqual(report["relative_l1"], 0.1)
        self.assertAlmostEqual(report["relative_l2"], 0.1)
        self.assertFalse(report["used_for_stopping"])
        self.assertIsNone(field_metrics([0, 0], [1, 1])["relative_l2"])
        for reference, prediction in [([], []), ([1], [1, 2]), ([1], [float("nan")])]:
            with self.assertRaises(ValueError):
                field_metrics(reference, prediction)


if __name__ == "__main__":
    unittest.main(verbosity=2)
