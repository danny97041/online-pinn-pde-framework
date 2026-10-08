"""Lazy scientific plots from stored metrics and selected checkpoint inference."""


def v3_visualize(
    root,
    equation,
    arm,
    seed,
    kind="comparison",
    resolution=32,
    time_fraction=0.5,
    component=0,
    predictor=None,
):
    if kind not in {"comparison", "field", "history"}:
        raise ValueError("Unknown visualization")
    if equation not in U_PROBLEMS or arm not in U_ARMS or type(seed) is not int:
        raise ValueError("Unknown model identity")
    root = Path(root)
    registry = V3ModelRegistry(root)
    summary = registry.summary
    directory = root / "visualizations"
    directory.mkdir(exist_ok=True)
    import matplotlib.pyplot as plt

    paths, details = [], {}

    def save(fig, name):
        fig.tight_layout()
        for suffix in ["png", "pdf"]:
            path = directory / (name + "." + suffix)
            fig.savefig(path, dpi=140, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)

    if kind == "comparison":
        rows = [r for r in summary["aggregate"] if r["equation"] == equation]
        if not rows:
            raise ValueError("No stored results for equation")
        means = [r["field_l2_mean"] * 100 for r in rows]
        spreads = [(r.get("field_l2_sample_std") or 0) * 100 for r in rows]
        fig, axis = plt.subplots(figsize=(9, 4))
        axis.bar(range(len(rows)), means, yerr=spreads, capsize=4, color="#3974bd")
        for i, r in enumerate(rows):
            values = [
                v["field_l2"] * 100
                for v in summary["rows"]
                if v["equation"] == equation and v["arm"] == r["arm"]
            ]
            axis.scatter(
                np.full(len(values), i), values, color="#df7b36", zorder=3, s=25
            )
        axis.set_xticks(
            range(len(rows)), [r["arm"] for r in rows], rotation=15, ha="right"
        )
        axis.set_ylabel("Selected field L2 (%)")
        axis.set_title(equation + " / mean, sample SD and individual seeds")
        save(fig, equation + "_comparison")
        equations = [
            eq
            for eq in U_PROBLEMS
            if any(r["equation"] == eq for r in summary["aggregate"])
        ]
        matrix = np.full((len(equations), len(U_ARMS)), np.nan)
        for r in summary["aggregate"]:
            matrix[equations.index(r["equation"]), U_ARMS.index(r["arm"])] = (
                r["field_l2_mean"] * 100
            )
        fig, axis = plt.subplots(figsize=(10, max(3, len(equations) * 0.52)))
        image = axis.imshow(
            np.ma.masked_invalid(np.log10(np.maximum(matrix, 1e-8))),
            cmap="viridis_r",
            aspect="auto",
        )
        axis.set_xticks(range(len(U_ARMS)), U_ARMS, rotation=15, ha="right")
        axis.set_yticks(range(len(equations)), equations)
        for i, j in np.ndindex(matrix.shape):
            if np.isfinite(matrix[i, j]):
                axis.text(
                    j,
                    i,
                    f"{matrix[i, j]:.3g}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
                )
        fig.colorbar(image, ax=axis, label="log10 field L2 (%)")
        save(fig, "equation_method_heatmap")
        details = {
            "error_bars": "sample standard deviation; absent for one seed",
            "optimizer_updates": 0,
        }
    elif kind == "field":
        if (
            type(resolution) is not int
            or not 8 <= resolution <= 64
            or not 0 <= time_fraction <= 1
        ):
            raise ValueError("Grid must be 8..64; time fraction must be 0..1")
        info = registry.info(equation, arm, seed)
        if type(component) is not int or not 0 <= component < info["output_dimension"]:
            raise ValueError("Output component outside model dimensions")
        d = np.asarray(info["domain"], float)
        xx, yy = np.meshgrid(
            np.linspace(*d[0], resolution), np.linspace(*d[1], resolution)
        )
        points = np.column_stack([xx.ravel(), yy.ravel()])
        if len(d) == 3:
            points = np.column_stack(
                [
                    points,
                    np.full(len(points), d[2, 0] + time_fraction * (d[2, 1] - d[2, 0])),
                ]
            )
        predictor = predictor or release_predictor(root)
        prediction = np.concatenate(
            [
                np.asarray(
                    predictor(
                        equation, arm, seed, points[start : start + 1024].tolist()
                    )["prediction"],
                    float,
                )
                for start in range(0, len(points), 1024)
            ]
        )
        if equation == "wave2d":
            e = registry.wave_config()["equation"]
            kx, ky = np.pi * e["kx"], np.pi * e["ky"]
            omega = e["wave_speed"] * np.sqrt(e["lambda_target"] * (kx * kx + ky * ky))
            exact = (
                np.sin(kx * points[:, 0])
                * np.sin(ky * points[:, 1])
                * np.cos(omega * points[:, 2])
            )[:, None]
        else:
            exact = candidate_field(equation, points)
        if prediction.shape != exact.shape or not np.isfinite(prediction).all():
            raise ValueError("Prediction grid shape mismatch")
        values = [
            exact[:, component],
            prediction[:, component],
            np.abs(prediction[:, component] - exact[:, component]),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        low, high = min(values[0].min(), values[1].min()), max(
            values[0].max(), values[1].max()
        )
        for i, (axis, value, title) in enumerate(
            zip(axes, values, ["Exact", "Selected prediction", "Absolute error"])
        ):
            im = axis.imshow(
                value.reshape(xx.shape),
                origin="lower",
                extent=[*d[0], *d[1]],
                aspect="auto",
                cmap="viridis",
                **({"vmin": low, "vmax": high} if i < 2 else {"vmin": 0}),
            )
            axis.set_title(title)
            axis.set_xlabel(info["coordinates"][0])
            axis.set_ylabel(info["coordinates"][1])
            fig.colorbar(im, ax=axis)
        fig.suptitle(f"{equation} / {arm} / seed {seed} / {info['outputs'][component]}")
        save(fig, f"{equation}_{arm}_{seed}_field")
        details = {
            "points": len(points),
            "coordinates": info["coordinates"],
            "time_fraction": time_fraction if len(d) == 3 else None,
            "component": info["outputs"][component],
            "stage": "selected",
            "new_visualization_grid": True,
            "optimizer_updates": 0,
        }
    else:
        registry.info(equation, arm, seed)
        if equation == "wave2d":
            member = f"trials/seed_{seed}_{arm}/result.json"
            parent = root / "inputs/wave_parent.zip"
            if parent.exists():
                with zipfile.ZipFile(parent) as archive:
                    result = json.loads(archive.read(member))
            else:
                result = json.loads((root / member).read_text(encoding="utf-8"))
        else:
            result = json.loads(
                (
                    root / f"candidates/{equation}/seed_{seed}_{arm}/result.json"
                ).read_text(encoding="utf-8")
            )
        rows = []
        for item in result.get("history", []):
            value = item.get(
                "validation_l2", item.get("validation_metrics", {}).get("relative_l2")
            )
            if value is not None:
                rows.append((item["iteration"], value * 100))
        if not rows:
            raise ValueError("No stored validation L2 history for this model")
        fig, axis = plt.subplots(figsize=(9, 4))
        axis.semilogy(*np.asarray(rows).T, color="#3974bd")
        axis.set_xlabel("Adam updates")
        axis.set_ylabel("Fixed validation L2 (%)")
        axis.set_title(f"{equation} / {arm} / seed {seed}")
        save(fig, f"{equation}_{arm}_{seed}_history")
        details = {"stored_history_points": len(rows), "optimizer_updates": 0}
    report = {
        "kind": kind,
        "equation": equation,
        "files": [p.relative_to(root).as_posix() for p in paths],
        "details": details,
        "training_executed": False,
    }
    u_json(directory / "latest.json", report)
    return report
