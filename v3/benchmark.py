"""Optional four-seed orchestration; scalar training contracts stay unchanged."""

V3_BENCHMARK_SEEDS = [3234, 3235, 3236, 3237]


def release_training_benchmark(root, settings, destination, output_zip):
    import shutil

    if settings["execution"] != "new":
        raise ValueError(
            "Four-seed mode starts new trials; resume completed seeds individually"
        )
    settings = release_validate_settings(settings)
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Choose a new benchmark directory")
    destination.mkdir(parents=True)
    rows = {}
    u_json(destination / "settings.json", {**settings, "seed": V3_BENCHMARK_SEEDS[0]})
    u_json(
        destination / "benchmark_settings.json",
        {
            "seeds": V3_BENCHMARK_SEEDS,
            "settings": settings,
            "resume": "select a completed seed individually",
        },
    )
    status = "running"

    def collect(child, summary):
        for row in summary["rows"]:
            rows[(row["equation"], row["arm"], row["seed"])] = row
        # Only a preparation/completed-trial boundary reaches this callback.
        for name in ["candidates", "trials", "postprocess", "normalization", "inputs"]:
            if (child / name).exists():
                shutil.copytree(child / name, destination / name, dirs_exist_ok=True)
        for name in [
            "contract.json",
            "runtime.json",
            "runtime_versions.json",
            "wiring.json",
        ]:
            if (child / name).exists():
                shutil.copyfile(child / name, destination / name)
        contract_path = destination / "contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract.update(
            experiment="v3_additional_benchmark",
            policy={
                **release_training_policy(settings),
                "seeds": list(V3_BENCHMARK_SEEDS),
            },
        )
        u_json(contract_path, contract)
        normalization = child / "normalization.json"
        if normalization.exists():
            incoming = json.loads(normalization.read_text(encoding="utf-8"))
            combined_path = destination / "normalization.json"
            combined = (
                json.loads(combined_path.read_text(encoding="utf-8"))
                if combined_path.exists()
                else {"policy": incoming["policy"], "seeds": {}}
            )
            if combined["policy"] != incoming["policy"]:
                raise ValueError("Benchmark normalization policy mismatch")
            combined["seeds"].update(incoming["seeds"])
            u_json(combined_path, combined)
        u_report(destination, list(rows.values()), {"report": "complete"}, {})
        report = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
        report.update(
            status=status,
            benchmark_seeds=list(V3_BENCHMARK_SEEDS),
            policy={
                **release_training_policy(settings),
                "seeds": list(V3_BENCHMARK_SEEDS),
            },
        )
        u_json(destination / "summary.json", report)
        u_pack(destination, output_zip)

    for seed in V3_BENCHMARK_SEEDS:
        child = destination.parent / (destination.name + "_seed_" + str(seed))
        single = {**settings, "seed": seed}
        print("4시드 실험:", seed, "/", V3_BENCHMARK_SEEDS)
        release_training(
            root,
            single,
            child,
            output_zip=child.with_suffix(".zip"),
            completed_boundary=collect,
        )
    status = "complete"
    collect(child, json.loads((child / "summary.json").read_text(encoding="utf-8")))
    return Path(output_zip)
