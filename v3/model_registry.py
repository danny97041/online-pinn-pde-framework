"""Saved-model catalogue and explicitly approved inference selection.

Registry entries select immutable checkpoints; they never replace trained files.
An input result archive is not modified. Registry state is part of a derived ZIP.
"""


class V3ModelRegistry:
    def __init__(self, root, predictor=None):
        self.root = Path(root)
        self.summary = json.loads(
            (self.root / "summary.json").read_text(encoding="utf-8")
        )
        self.predictor = predictor
        self.rows = {
            (r["equation"], r["arm"], r["seed"]): r for r in self.summary["rows"]
        }
        if len(self.rows) != len(self.summary["rows"]) or any(
            equation not in U_PROBLEMS
            or arm not in U_ARMS
            or type(seed) is not int
            or seed < 0
            for equation, arm, seed in self.rows
        ):
            raise ValueError("Invalid or duplicate model catalogue identity")
        self.path = self.root / "model_registry.json"
        self.state = (
            json.loads(self.path.read_text(encoding="utf-8"))
            if self.path.exists()
            else {"version": 1, "active": {}, "history": []}
        )
        if (
            self.state.get("version") != 1
            or not isinstance(self.state.get("active"), dict)
            or not isinstance(self.state.get("history"), list)
        ):
            raise ValueError("Invalid model registry")
        for equation, record in self.state["active"].items():
            if equation != record.get("equation"):
                raise ValueError("Registry equation mismatch")
            info = self.info(
                record["equation"], record["arm"], record["seed"], checksum=True
            )
            if not info["checkpoint_available"] or info[
                "checkpoint_sha256"
            ] != record.get("checkpoint_sha256"):
                raise ValueError("Registry checkpoint mismatch")

    def wave_config(self):
        with zipfile.ZipFile(self.root / "inputs/wave_source.zip") as archive:
            return json.loads(archive.read("config/wave2d_config.json"))

    def info(self, equation, arm, seed, checksum=False):
        if type(seed) is not int or (equation, arm, seed) not in self.rows:
            raise ValueError("No completed model for this equation/method/seed")
        nested = (
            equation == "wave2d" and (self.root / "inputs/wave_parent.zip").is_file()
        )
        if equation == "wave2d":
            config = self.wave_config()
            d = config["domain"]
            domain = [[d[k + "_min"], d[k + "_max"]] for k in ("x", "y", "t")]
            coordinates, outputs = ["x", "y", "t"], ["u"]
            member = f"postprocess/trials/seed_{seed}_{arm}/selected.weights.h5"
        else:
            spec = CANDIDATES[equation]
            domain, outputs = spec["domain"], spec["outputs"]
            coordinates = (
                ["x", "t"] if equation == "burgers" else ["x", "y", "t"][: len(domain)]
            )
            member = f"candidates/{equation}/seed_{seed}_{arm}/selected_weights.npz"
        digest = None
        if nested:
            with zipfile.ZipFile(self.root / "inputs/wave_parent.zip") as archive:
                available = member in archive.namelist()
                if available and checksum:
                    digest = hashlib.sha256(archive.read(member)).hexdigest()
        else:
            path = self.root / member
            available = path.is_file()
            if available and checksum:
                digest = u_sha(path)
        row = self.rows[(equation, arm, seed)]
        return {
            "equation": equation,
            "arm": arm,
            "seed": seed,
            "coordinates": coordinates,
            "domain": domain,
            "outputs": outputs,
            "output_dimension": len(outputs),
            "stage": "selected",
            "checkpoint_available": available,
            "checkpoint_sha256": digest,
            "checkpoint": {
                "archive": "inputs/wave_parent.zip" if nested else None,
                "member": member,
            },
            "field_l2": row["field_l2"],
        }

    def list_models(self, equation=None):
        if equation is not None and equation not in U_PROBLEMS:
            raise ValueError("Unknown equation")
        return [
            self.info(*key)
            for key in sorted(self.rows)
            if equation is None or key[0] == equation
        ]

    def _probe(self, record):
        if self.predictor is None:
            raise ValueError("Model predictor unavailable")
        info = self.info(
            record["equation"], record["arm"], record["seed"], checksum=True
        )
        if not info["checkpoint_available"]:
            raise FileNotFoundError("Saved checkpoint unavailable")
        if (
            record.get("checkpoint_sha256", info["checkpoint_sha256"])
            != info["checkpoint_sha256"]
        ):
            raise ValueError("Checkpoint changed since registration")
        d = np.asarray(info["domain"], float)
        points = ((d[:, 0] + d[:, 1]) / 2)[None, :].tolist()
        result = self.predictor(info["equation"], info["arm"], info["seed"], points)
        prediction = np.asarray(result["prediction"], float)
        if (
            prediction.shape != (1, info["output_dimension"])
            or not np.isfinite(prediction).all()
        ):
            raise ValueError("Saved-model loading probe failed")
        return {
            k: info[k]
            for k in ["equation", "arm", "seed", "checkpoint_sha256", "stage"]
        } | {"model_load_probe_passed": True}

    def activate(self, equation, arm, seed, approved=False):
        if approved is not True:
            raise ValueError("Explicit manual approval required")
        record = self._probe({"equation": equation, "arm": arm, "seed": seed})
        previous = self.state["active"].get(equation)
        self.state["history"].append(
            {
                "operation": "activate",
                "equation": equation,
                "previous": previous,
                "selected": record,
            }
        )
        self.state["active"][equation] = record
        u_json(self.path, self.state)
        return record

    def rollback(self, equation, approved=False):
        if approved is not True:
            raise ValueError("Explicit manual approval required")
        events = [
            e
            for e in self.state["history"]
            if e.get("equation") == equation
            and e.get("operation") in {"activate", "rollback"}
        ]
        if not events:
            raise ValueError("No previous registry selection")
        current = self.state["active"].get(equation)
        previous = events[-1].get("previous")
        if previous is not None:
            previous = self._probe(previous)
            self.state["active"][equation] = previous
        else:
            self.state["active"].pop(equation, None)
        self.state["history"].append(
            {
                "operation": "rollback",
                "equation": equation,
                "previous": current,
                "selected": previous,
            }
        )
        u_json(self.path, self.state)
        return previous

    def predict_active(self, equation, points):
        record = self.state["active"].get(equation)
        if record is None:
            raise ValueError("No manually selected inference model")
        info = self.info(equation, record["arm"], record["seed"], checksum=True)
        if info["checkpoint_sha256"] != record["checkpoint_sha256"]:
            raise ValueError("Registered checkpoint changed")
        return self.predictor(equation, record["arm"], record["seed"], points)
