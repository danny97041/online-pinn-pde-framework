"""Independent surface/volume quadrature for integral PDE balance diagnostics."""

import numpy as np
from .equations import EQUATIONS
from .storage import arrays_sha


def balance_report(runtime, resolution=8):
    name = runtime.settings.equation
    spec = EQUATIONS[name]
    bounds = np.asarray(runtime.source.domain, float)
    time_axis = spec["time_axis"]
    spatial = [a for a in range(len(bounds)) if a != time_axis]
    widths = np.ptp(bounds, axis=1)
    volume = float(np.prod(widths[spatial]))
    coefficient = float(runtime.coefficient)
    slices = (
        [None]
        if time_axis is None
        else bounds[time_axis, 0]
        + widths[time_axis]
        * np.array(
            [0.85, 0.925, 1.0]
            if runtime.settings.generalization == "time_extrapolation"
            else [0.2, 0.5, 0.8]
        )
    )
    axes = [
        bounds[a, 0] + widths[a] * (np.arange(resolution) + 0.5) / resolution
        for a in spatial
    ]
    space = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
        -1, len(spatial)
    )

    def points(values, time):
        x = np.zeros((len(values), len(bounds)))
        x[:, spatial] = values
        if time_axis is not None:
            x[:, time_axis] = time
        return x

    def terms(x):
        q, g, h = runtime.derivatives(x)
        q, g, h = q.numpy(), g.numpy(), h.numpy()
        f = runtime.source.forcing(x)
        if len(spec["outputs"]) == 3:
            flux = np.zeros((len(x), 3, 2))
            velocity = q[:, :2]
            for i in range(2):
                for j in range(2):
                    flux[:, i, j] = (
                        q[:, i] * velocity[:, j]
                        + (q[:, 2] if i == j else 0)
                        - coefficient * g[:, i, j]
                    )
            flux[:, 2, :] = velocity
            div = g[:, 0, 0] + g[:, 1, 1]
            bulk = -f.copy()
            for i in range(2):
                bulk[:, i] += (g[:, i, time_axis] if time_axis is not None else 0) - q[
                    :, i
                ] * div
            return flux, bulk
        if name == "elasticity2d":
            lam = coefficient * 0.25 / (1.25 * 0.5)
            mu = coefficient / 2.5
            trace = g[:, 0, 0] + g[:, 1, 1]
            stress = np.stack(
                [
                    np.stack(
                        [
                            lam * trace + 2 * mu * g[:, 0, 0],
                            mu * (g[:, 0, 1] + g[:, 1, 0]),
                        ],
                        axis=1,
                    ),
                    np.stack(
                        [
                            mu * (g[:, 0, 1] + g[:, 1, 0]),
                            lam * trace + 2 * mu * g[:, 1, 1],
                        ],
                        axis=1,
                    ),
                ],
                axis=1,
            )
            return -stress, -f
        flux = -coefficient * g[:, 0, spatial]
        bulk = -f[:, 0]
        if name == "poisson2d":
            flux = -g[:, 0, spatial]
            bulk = -coefficient * f[:, 0]
        elif name == "helmholtz":
            flux = -g[:, 0, spatial]
            bulk -= coefficient * q[:, 0]
        elif name == "darcy2d":
            flux *= (1 + 0.5 * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]))[
                :, None
            ]
        elif name == "burgers":
            flux[:, 0] += q[:, 0] ** 2 / 2
            bulk += g[:, 0, time_axis]
        elif name == "wave2d":
            bulk += h[:, 0, time_axis, time_axis]
        else:
            bulk += g[:, 0, time_axis]
            if name == "reaction_diffusion":
                bulk += -q[:, 0] + q[:, 0] ** 3
            if name == "advection_diffusion":
                flux += q[:, 0:1] * np.array([0.5, 0.25])
        return flux[:, None, :], bulk[:, None]

    before = runtime.snapshot()
    results = []
    all_coordinates = []
    for time in slices:
        x = points(space, time)
        _, bulk = terms(x)
        all_coordinates.append(x)
        integrated_bulk = volume * bulk.mean(axis=0)
        surface = np.zeros_like(integrated_bulk)
        for j, a in enumerate(spatial):
            other = [i for i in range(len(spatial)) if i != j]
            face = (
                space[::resolution]
                if len(spatial) == 2 and j == 1
                else space[:resolution] if len(spatial) == 2 else space[:1]
            )
            for end, sign in ((0, -1), (1, 1)):
                face = face.copy()
                face[:, j] = bounds[a, end]
                if other:
                    face[:, other[0]] = axes[other[0]]
                fx = points(face, time)
                flux, _ = terms(fx)
                all_coordinates.append(fx)
                surface += sign * volume / widths[a] * flux[:, :, j].mean(axis=0)
        defect = integrated_bulk + surface
        results.append(
            {
                "time": None if time is None else float(time),
                "bulk_integral": integrated_bulk.tolist(),
                "outgoing_flux_integral": surface.tolist(),
                "balance_defect": defect.tolist(),
                "normalized_balance_defect": (
                    defect / (np.asarray(runtime.scales["residual"]) * volume)
                ).tolist(),
            }
        )
    runtime.assert_unchanged(before)
    return {
        "definition": "volume source/time terms + separately sampled outgoing face flux; midpoint quadrature",
        "spatial_region": "full rectangular spatial domain; interpolation field holdout is reported separately",
        "resolution_per_axis": resolution,
        "coordinates_sha": arrays_sha({"points": np.concatenate(all_coordinates)}),
        "slices": results,
        "used_for_training_or_stopping": False,
        "state_guard": "PASS",
    }
