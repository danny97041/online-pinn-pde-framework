"""Independent interpolation and temporal-extrapolation dataset contracts.

No TensorFlow, model loading, label generation or training occurs here. The V4
runner must use these contracts on every initial batch and resampling cycle.
"""

from dataclasses import dataclass
import hashlib
import json

import numpy as np

MODES = {"basic", "interpolation", "time_extrapolation"}
FAMILIES = {"observations", "pde", "boundary", "initial", "gauge"}
SEED_ORIGIN = 3234
BLOCK_COUNT = 16


@dataclass(frozen=True)
class GeneralizationPlan:
    """Immutable split, shared by methods and independent of sampling cycles.

    Spatial block numbering is row-major: x changes fastest in 2D. Edges belong
    to the block on their positive side; the domain maximum is in the last block.
    Known initial/boundary/gauge constraints remain available in interpolation.
    """

    domain: tuple
    spatial_axes: tuple
    time_axis: int | None = None
    mode: str = "basic"
    seed: int = SEED_ORIGIN
    block_override: int | None = None
    time_train_fraction: float = 0.8
    version: str = "v4-generalization-16-v1"

    def __post_init__(self):
        domain = np.asarray(self.domain, dtype=np.float64)
        if domain.ndim != 2 or domain.shape[1] != 2 or not len(domain):
            raise ValueError("domain must contain [minimum, maximum] per axis")
        if not np.isfinite(domain).all() or np.any(domain[:, 0] >= domain[:, 1]):
            raise ValueError("Domain limits must be finite and increasing")
        axes = tuple(self.spatial_axes)
        if len(axes) not in (1, 2) or any(type(a) is not int for a in axes):
            raise ValueError("Exactly one or two integer spatial axes are supported")
        if len(set(axes)) != len(axes) or any(a < 0 or a >= len(domain) for a in axes):
            raise ValueError("Invalid spatial axes")
        if self.time_axis is not None and (
            type(self.time_axis) is not int
            or self.time_axis in axes
            or self.time_axis < 0
            or self.time_axis >= len(domain)
        ):
            raise ValueError("Invalid time axis")
        covered = set(axes) | (
            {self.time_axis} if self.time_axis is not None else set()
        )
        if covered != set(range(len(domain))):
            raise ValueError("Every coordinate axis must be spatial or temporal")
        if self.mode not in MODES or type(self.seed) is not int or self.seed < 0:
            raise ValueError("Invalid mode or seed")
        if self.version != "v4-generalization-16-v1":
            raise ValueError("Unsupported split version")
        if self.block_override is not None and (
            type(self.block_override) is not int or not 0 <= self.block_override < 16
        ):
            raise ValueError("block_override must be an integer in 0..15")
        if (
            not np.isfinite(self.time_train_fraction)
            or not 0 < self.time_train_fraction < 1
        ):
            raise ValueError("time_train_fraction must be between zero and one")
        if self.mode == "time_extrapolation" and self.time_axis is None:
            raise ValueError(
                "Temporal extrapolation requires a time-dependent equation"
            )
        object.__setattr__(self, "domain", tuple(tuple(map(float, r)) for r in domain))
        object.__setattr__(self, "spatial_axes", axes)

    @property
    def block_id(self):
        if self.block_override is not None:
            return self.block_override
        # Coprime stride visits every block once over 16 consecutive seeds.
        return (5 * (self.seed - SEED_ORIGIN) + 7) % BLOCK_COUNT

    @property
    def cutoff(self):
        if self.time_axis is None:
            return None
        lo, hi = self.domain[self.time_axis]
        return lo + self.time_train_fraction * (hi - lo)

    @property
    def training_domain(self):
        """Sampling bounds for LHS, RAD candidate generation and BC sampling."""
        bounds = [list(row) for row in self.domain]
        if self.mode == "time_extrapolation":
            bounds[self.time_axis][1] = self.cutoff
        return bounds

    def coordinates(self, values):
        points = np.asarray(values, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != len(self.domain):
            raise ValueError("Coordinates must be an N by domain-dimension array")
        bounds = np.asarray(self.domain)
        if not np.isfinite(points).all():
            raise ValueError("Coordinates must be finite")
        if np.any(points < bounds[:, 0]) or np.any(points > bounds[:, 1]):
            raise ValueError("Coordinates outside the declared domain")
        return points

    def spatial_block_ids(self, values):
        points = self.coordinates(values)
        divisions = 16 if len(self.spatial_axes) == 1 else 4
        ids = np.zeros(len(points), dtype=np.int64)
        for position, axis in enumerate(self.spatial_axes):
            lo, hi = self.domain[axis]
            unit = (points[:, axis] - lo) / (hi - lo)
            cells = np.minimum(
                np.floor(unit * divisions).astype(np.int64), divisions - 1
            )
            ids += cells * divisions**position
        return ids

    def training_mask(self, values, family="observations"):
        """Apply on every renewal. Initial/BC constraints are not held-out labels."""
        if family not in FAMILIES:
            raise ValueError("Unknown training family")
        points = self.coordinates(values)
        if self.mode == "time_extrapolation":
            return points[:, self.time_axis] <= self.cutoff
        if self.mode == "interpolation" and family == "observations":
            return self.spatial_block_ids(points) != self.block_id
        return np.ones(len(points), dtype=bool)

    def evaluation_mask(self, values):
        points = self.coordinates(values)
        if self.mode == "interpolation":
            return self.spatial_block_ids(points) == self.block_id
        if self.mode == "time_extrapolation":
            return points[:, self.time_axis] > self.cutoff
        return np.ones(len(points), dtype=bool)

    def split_observations(self, coordinates, labels):
        """Return aligned copies; never discard or alter the source dataset."""
        points = self.coordinates(coordinates)
        labels = np.asarray(labels)
        if labels.ndim not in (1, 2) or len(labels) != len(points):
            raise ValueError("Labels must align with coordinates")
        if not np.issubdtype(labels.dtype, np.number) or not np.isfinite(labels).all():
            raise ValueError("Labels must be finite numeric values")
        mask = self.training_mask(points)
        if not mask.any():
            raise ValueError("No training observations remain")
        if self.mode != "basic" and mask.all():
            raise ValueError(
                "No held-out observations exist; revise the observation layout"
            )
        return {
            "train_coordinates": points[mask].copy(),
            "train_labels": labels[mask].copy(),
            "heldout_coordinates": points[~mask].copy(),
            "heldout_labels": labels[~mask].copy(),
            "train_mask": mask.copy(),
        }

    def assert_training(self, coordinates, family="observations"):
        """Fail closed if any training/resampled point violates the split."""
        if not self.training_mask(coordinates, family).all():
            raise ValueError(
                "Training point leaks into the held-out observation/time region"
            )

    def validation_mask(self, coordinates, family="observations"):
        """Stopping/tuning validation uses only the permitted training region."""
        return self.training_mask(coordinates, family)

    def fingerprint(self):
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_dict(self):
        return {
            "version": self.version,
            "domain": [list(row) for row in self.domain],
            "spatial_axes": list(self.spatial_axes),
            "time_axis": self.time_axis,
            "mode": self.mode,
            "seed": self.seed,
            "block_override": self.block_override,
            "time_train_fraction": float(self.time_train_fraction),
        }

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    def summary(self):
        return {
            **self.to_dict(),
            "fingerprint": self.fingerprint(),
            "block_count": BLOCK_COUNT if self.mode == "interpolation" else None,
            "heldout_block": self.block_id if self.mode == "interpolation" else None,
            "training_domain": self.training_domain,
            "time_cutoff": self.cutoff if self.mode == "time_extrapolation" else None,
            "heldout_used_for_stopping": False,
            "independent_training_required": self.mode != "basic",
        }

    def fixed_evaluation_points(self, per_spatial_axis=32, time_count=21):
        """Deterministic cell-centre grid; never uses or advances training RNG.

        Interpolation grids require multiples of 4 (2D) or 16 (1D), ensuring
        every block has the same number of evaluation points.
        """
        divisions = 16 if len(self.spatial_axes) == 1 else 4
        if type(per_spatial_axis) is not int or per_spatial_axis < divisions:
            raise ValueError("Evaluation resolution is too small")
        if self.mode == "interpolation" and per_spatial_axis % divisions:
            raise ValueError(
                "Interpolation resolution must be divisible by block divisions"
            )
        if type(time_count) is not int or time_count < 1:
            raise ValueError("time_count must be positive")
        axes = []
        for axis, (lo, hi) in enumerate(self.domain):
            if axis == self.time_axis:
                start = self.cutoff if self.mode == "time_extrapolation" else lo
                values = start + (np.arange(time_count) + 0.5) / time_count * (
                    hi - start
                )
            else:
                values = lo + (np.arange(per_spatial_axis) + 0.5) / per_spatial_axis * (
                    hi - lo
                )
            axes.append(values)
        if np.prod([len(a) for a in axes], dtype=np.int64) > 1_000_000:
            raise ValueError("Evaluation grid exceeds one million points")
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
            -1, len(axes)
        )
        selected = points[self.evaluation_mask(points)]
        if not len(selected):
            raise ValueError("Empty evaluation region")
        return selected.copy()


def field_metrics(reference, prediction):
    """Aggregate relative errors for final reporting, not stopping decisions.

    Returns null relative errors for an all-zero reference instead of adding an
    arbitrary denominator epsilon. Components should also be reported separately
    by the runner when output dimensions have different meanings/scales.
    """
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if (
        reference.shape != prediction.shape
        or reference.ndim not in (1, 2)
        or not reference.size
    ):
        raise ValueError("Reference and prediction need matching non-empty shapes")
    if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        raise ValueError("Reference and prediction must be finite")
    error = prediction - reference
    l1_denominator = float(np.abs(reference).sum())
    l2_denominator = float(np.linalg.norm(reference))
    return {
        "relative_l1": (
            float(np.abs(error).sum() / l1_denominator) if l1_denominator else None
        ),
        "relative_l2": (
            float(np.linalg.norm(error) / l2_denominator) if l2_denominator else None
        ),
        "mse": float(np.mean(error**2)),
        "relative_errors_defined": bool(l1_denominator and l2_denominator),
        "used_for_stopping": False,
    }
