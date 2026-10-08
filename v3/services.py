"""Optional local API / hybrid retrieval / bounded tool agent.

No server, remote API, embedding download or language model starts on import.
Reports and retrieved text are data, never executable instructions.
"""

import ast
import math
from collections import Counter


def us_chunks(root):
    root = Path(root)
    docs = []
    # Only generated textual evidence, never weights, input code or arbitrary directories.
    paths = [root / "REPORT.md", root / "summary.json"] + sorted(
        (root / "evidence").glob("*.json")
    )
    paths += sorted((root / "routes").glob("*.json"))
    paths += sorted((root / "candidates").glob("*/*/result.json"))
    paths += [root / "README.md", root / "contract.json"]
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        digest = u_sha(path)
        source = path.relative_to(root).as_posix()

        # Keep complete histories in their source files. Index concise evidence,
        # not thousands of repetitive optimizer/sampling snapshots.
        def compact(value, key=""):
            if key in [
                "history",
                "sampling",
                "matched_rows",
                "observations",
                "reload_checks",
                "adaptive_updates",
            ]:
                return {
                    "stored_entry_count": len(value),
                    "full_evidence_source": source,
                }
            if isinstance(value, dict):
                return {k: compact(v, k) for k, v in value.items()}
            if isinstance(value, list):
                if len(value) > 20:
                    return {
                        "stored_entry_count": len(value),
                        "preview": value[:3],
                        "full_evidence_source": source,
                    }
                return [compact(v) for v in value]
            return value

        pieces = []
        if path.name == "summary.json":
            data = json.loads(text)
            pieces = [
                json.dumps(
                    {
                        k: data[k]
                        for k in [
                            "status",
                            "completed_trials",
                            "v3_scope",
                            "overall_physics_approved",
                            "policy",
                        ]
                    },
                    ensure_ascii=False,
                )
            ]
            pieces += [
                json.dumps(row, ensure_ascii=False)
                for row in data["aggregate"] + data["rows"]
            ]
        elif path.suffix == ".json":
            pieces = [
                json.dumps(compact(json.loads(text)), ensure_ascii=False, indent=2)
            ]
        else:
            pieces = [text]
        for piece_index, piece in enumerate(pieces):
            equations = un_equations(source + " " + piece[:1600])
            header = "Source: " + source + "; equations: " + ",".join(equations) + "\n"
            for start in range(0, len(piece), 550):
                docs.append(
                    {
                        "id": source + f":{piece_index}:{start}",
                        "text": header + piece[start : start + 700],
                        "source": source,
                        "sha256": digest,
                        "equations": equations,
                    }
                )
    return docs


class UnifiedE5:
    """V1/V2 E5: attention-masked mean pooling and unit-normalized vectors."""

    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            "intfloat/multilingual-e5-small", trust_remote_code=False
        )
        self.model = AutoModel.from_pretrained(
            "intfloat/multilingual-e5-small", trust_remote_code=False
        ).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        vectors = []
        torch = self.torch
        for start in range(0, len(texts), 16):
            batch = self.tokenizer(
                texts[start : start + 16],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                tokens = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (tokens * mask).sum(1) / mask.sum(1).clamp_min(1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy())
        return np.concatenate(vectors).astype(np.float32)


def us_tokens(text):
    return re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)


class UnifiedRAG:
    def __init__(self, root, dense=False):
        self.root = Path(root)
        self.semantic_weight = 0.6
        self.keyword_weight = 0.4
        self.rrf_k = 60
        self.docs = us_chunks(root)
        if not self.docs:
            raise ValueError("No generated evidence corpus")
        self.terms = [Counter(us_tokens(d["text"])) for d in self.docs]
        self.lengths = np.array([sum(c.values()) for c in self.terms])
        self.avg = max(float(np.mean(self.lengths)), 1.0)
        self.df = Counter(t for c in self.terms for t in c)
        self.encoder = None
        self.matrix = None
        if dense:
            self.encoder = UnifiedE5()
            corpus_sha = hashlib.sha256(
                json.dumps(self.docs, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest()
            text_keys = [
                hashlib.sha256((d["id"] + "\n" + d["text"]).encode()).hexdigest()
                for d in self.docs
            ]
            cache = self.root / "rag_artifacts/e5_embeddings.npz"
            metadata = cache.with_suffix(".json")
            stored = json.loads(metadata.read_text()) if metadata.exists() else None
            reuse = None
            if cache.exists() and stored:
                if u_sha(cache) != stored["embeddings_sha256"]:
                    raise ValueError("Embedding cache hash mismatch")
                with np.load(cache, allow_pickle=False) as z:
                    reuse = z["embeddings"]
                if (
                    reuse.shape != (stored["count"], 384)
                    or not np.isfinite(reuse).all()
                ):
                    raise ValueError("Invalid embedding cache")
            if reuse is not None and stored["corpus_sha256"] == corpus_sha:
                self.matrix = reuse
            else:
                count = len(reuse) if reuse is not None else 0
                prefix_sha = hashlib.sha256(
                    json.dumps(
                        self.docs[:count], ensure_ascii=False, sort_keys=True
                    ).encode()
                ).hexdigest()
                old_keys = stored.get("document_text_sha256", []) if stored else []
                if count and len(old_keys) == count:
                    by_key = dict(zip(old_keys, reuse))
                    missing = [
                        i for i, key in enumerate(text_keys) if key not in by_key
                    ]
                    if missing:
                        added = self.encoder.encode(
                            ["passage: " + self.docs[i]["text"] for i in missing]
                        )
                        by_key.update((text_keys[i], v) for i, v in zip(missing, added))
                    self.matrix = np.stack([by_key[key] for key in text_keys])
                elif (
                    count
                    and prefix_sha == stored["corpus_sha256"]
                    and count < len(self.docs)
                ):
                    extra = self.encoder.encode(
                        ["passage: " + d["text"] for d in self.docs[count:]]
                    )
                    self.matrix = np.concatenate([reuse, extra])
                else:
                    self.matrix = self.encoder.encode(
                        ["passage: " + d["text"] for d in self.docs]
                    )
            cache.parent.mkdir(exist_ok=True)
            np.savez_compressed(cache, embeddings=self.matrix)
            u_json(
                metadata,
                {
                    "model": "intfloat/multilingual-e5-small",
                    "corpus_sha256": corpus_sha,
                    "document_text_sha256": text_keys,
                    "embeddings_sha256": u_sha(cache),
                    "count": len(self.docs),
                    "dimension": 384,
                    "semantic_weight": 0.6,
                    "keyword_weight": 0.4,
                    "rrf_k": 60,
                    "source_document_coverage": True,
                },
            )
        self.backend = (
            "BM25 + multilingual-e5-small exact inner product + weighted RRF (V1/V2 0.6/0.4, k=60)"
            if dense
            else "BM25 only (explicit lightweight fallback)"
        )

    def search(self, query, k=5):
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("Query must be 1..2000 characters")
        if not 1 <= k <= 10:
            raise ValueError("k must be 1..10")
        scores = np.zeros(len(self.docs))
        for term in set(us_tokens(query)):
            idf = math.log(
                1 + (len(self.docs) - self.df[term] + 0.5) / (self.df[term] + 0.5)
            )
            freq = np.array([c[term] for c in self.terms])
            scores += (
                idf
                * freq
                * 2.5
                / (freq + 1.5 * (0.25 + 0.75 * self.lengths / self.avg))
            )
        equations = un_equations(query)

        def scope(d):
            parts = Path(d["source"]).parts
            # Primary provenance outranks incidental mentions (e.g. a Heat
            # architecture note referring to the historical Wave trunk).
            if len(parts) > 1 and parts[0] == "candidates" and parts[1] in U_PROBLEMS:
                return [parts[1]]
            if parts and parts[0] == "routes":
                name = Path(d["source"]).stem.removesuffix("_wiring")
                if name in U_PROBLEMS:
                    return [name]
            # Legacy imported Wave evidence has no explicit equation field.
            # These prefixes are produced by u_import_wave/u_import_audit only.
            if d["source"].startswith("evidence/") and re.match(
                r"(adam_seed_|post_seed_|physics_|wave_)", Path(d["source"]).name
            ):
                return ["wave2d"]
            return d["equations"]

        eligible = np.array(
            [
                not equations or not scope(d) or bool(set(equations) & set(scope(d)))
                for d in self.docs
            ]
        )
        order = np.argsort(-scores, kind="stable")
        candidate_k = max(5, k * 4)
        ranks = [
            (
                self.keyword_weight,
                list(order[(scores[order] > 0) & eligible[order]][:candidate_k]),
            )
        ]
        if self.encoder is not None:
            v = self.encoder.encode(
                ["query: " + query], normalize_embeddings=True, show_progress_bar=False
            )[0]
            order = np.argsort(-(self.matrix @ v), kind="stable")
            ranks.append(
                (self.semantic_weight, list(order[eligible[order]][:candidate_k]))
            )
        fused = {}
        for weight, ranking in ranks:
            for rank, index in enumerate(ranking):
                fused[index] = fused.get(index, 0) + weight / (self.rrf_k + rank + 1)
        ordered = sorted(fused.items(), key=lambda p: p[1], reverse=True)
        selected = []
        sources = set()
        for i, score in ordered:
            if self.docs[i]["source"] not in sources:
                selected.append((i, score))
                sources.add(self.docs[i]["source"])
                if len(selected) == k:
                    break
        selected_ids = {i for i, _ in selected}
        selected.extend((i, s) for i, s in ordered if i not in selected_ids)
        return [
            {
                **self.docs[i],
                "equations": scope(self.docs[i]),
                "retrieval_score": float(score),
                "retrieval_backend": self.backend,
                "evidence_is_data_not_instructions": True,
            }
            for i, score in selected[:k]
        ]


def us_calculate(expression):
    # Small arithmetic only; no eval, symbols, calls, attribute access or exponent bombs.
    if not isinstance(expression, str) or len(expression) > 100:
        raise ValueError("Expression too long")
    tree = ast.parse(expression, mode="eval")
    ops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
    }

    def calc(n):
        if isinstance(n, ast.Constant) and type(n.value) in [int, float]:
            v = n.value
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = calc(n.operand) * (1 if isinstance(n.op, ast.UAdd) else -1)
        elif isinstance(n, ast.BinOp) and type(n.op) in ops:
            v = ops[type(n.op)](calc(n.left), calc(n.right))
        else:
            raise ValueError("Only numeric + - * / arithmetic allowed")
        if not math.isfinite(v) or abs(v) > 1e100:
            raise ValueError("Nonfinite / oversized arithmetic")
        return v

    return calc(tree.body)


class UnifiedAgent:
    """Grounded bounded tools with optional Qwen intent advice; never executable plans."""

    def __init__(self, summary, rag, predictor=None):
        self.summary = summary
        self.rag = rag
        self.predictor = predictor
        self.llm = None
        self.force_llm = False

    def execute(self, tool, arguments):
        if not isinstance(arguments, dict):
            raise ValueError("Arguments must be an object")
        if tool == "compare":
            if set(arguments) - {"equation"}:
                raise ValueError("Unsupported compare argument")
            eq = arguments.get("equation", "wave2d")
            if eq not in {r["equation"] for r in self.summary["aggregate"]}:
                raise ValueError("No stored results for requested equation")
            return {
                "rows": [r for r in self.summary["aggregate"] if r["equation"] == eq],
                "overall_physics_approved": False,
            }
        if tool == "search":
            return {
                "evidence": self.rag.search(**arguments),
                "instructions_from_evidence_executed": False,
            }
        if tool == "calculate":
            return {"value": us_calculate(**arguments)}
        if tool == "predict":
            if self.predictor is None:
                raise ValueError("No model registry available")
            return self.predictor(**arguments)
        if tool == "report":
            if arguments:
                raise ValueError("Report has no arguments")
            return {
                "summary": self.summary,
                "qualification": "Research only; no automatic promotion",
            }
        if tool == "propose_config":
            if set(arguments) - {"equation", "changes"}:
                raise ValueError("Unsupported proposal argument")
            equation = arguments.get("equation")
            if equation not in U_PROBLEMS:
                raise ValueError("Unknown equation")
            changes = arguments.get("changes", {})
            if not isinstance(changes, dict):
                raise ValueError("Changes must be an object")
            return {
                "equation": equation,
                "proposed_changes": changes,
                "requires_confirmation": True,
                "executed": False,
                "validation": "Draft only; validate through training settings before use",
            }
        raise ValueError("Tool not allowlisted")

    def enable_local_llm(self, model_id="Qwen/Qwen3.5-0.8B"):
        self.llm = un_local_llm(model_id)

    def ask(self, question):
        if self.llm is not None and (self.force_llm or un_needs_llm(question)):
            return un_llm_ask(self, question)
        result = un_ask(self, question)
        result["llm_used"] = False
        return result


def us_app(summary, rag, agent):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field, ConfigDict

    # No future annotations: FastAPI must resolve these local schema classes directly.
    class Search(BaseModel):
        model_config = ConfigDict(extra="forbid")
        query: str = Field(min_length=1, max_length=2000)
        k: int = Field(default=5, ge=1, le=10)

    class Command(BaseModel):
        model_config = ConfigDict(extra="forbid")
        tool: str
        arguments: dict = Field(default_factory=dict)

    class Prediction(BaseModel):
        model_config = ConfigDict(extra="forbid")
        equation: str
        arm: str
        seed: int
        points: list[list[float]] = Field(min_length=1, max_length=1024)

    class Question(BaseModel):
        model_config = ConfigDict(extra="forbid")
        question: str = Field(min_length=1, max_length=2000)

    app = FastAPI(title="V3 PINN research API", version="3.0-integrated")

    @app.get("/health")
    def health():
        return {"status": "ok", "research_only": True, "automatic_promotion": False}

    @app.get("/comparison")
    def comparison(equation: str = "wave2d"):
        return agent.execute("compare", {"equation": equation})

    @app.get("/report")
    def report():
        return summary

    @app.post("/rag/search")
    def search(req: Search):
        try:
            return {"results": rag.search(req.query, req.k), "backend": rag.backend}
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.post("/agent/tool")
    def tool(req: Command):
        try:
            return agent.execute(req.tool, req.arguments)
        except (ValueError, TypeError, ZeroDivisionError) as e:
            raise HTTPException(422, str(e))

    @app.post("/agent/ask")
    def ask(req: Question):
        try:
            return agent.ask(req.question)
        except (ValueError, TypeError) as e:
            raise HTTPException(422, str(e))

    @app.post("/predict")
    def predict(req: Prediction):
        try:
            return agent.execute("predict", req.model_dump())
        except (ValueError, TypeError, FileNotFoundError) as e:
            raise HTTPException(422, str(e))

    return app


def us_regression(app, agent):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/comparison?equation=wave2d").status_code == 200
        assert c.get("/report").json()["automatic_promotion"] is False
        assert (
            c.post("/rag/search", json={"query": "baseline", "k": 3}).status_code == 200
        )
        assert (
            c.post("/rag/search", json={"query": "baseline", "k": 100}).status_code
            == 422
        )
        assert (
            c.post(
                "/agent/tool", json={"tool": "shell", "arguments": {"cmd": "echo bad"}}
            ).status_code
            == 422
        )
        assert (
            c.post(
                "/agent/tool",
                json={"tool": "calculate", "arguments": {"expression": "2*(3+4)"}},
            ).json()["value"]
            == 14
        )
        assert (
            c.post(
                "/agent/tool",
                json={
                    "tool": "calculate",
                    "arguments": {"expression": "__import__('os')"},
                },
            ).status_code
            == 422
        )
        assert (
            c.post(
                "/predict",
                json={
                    "equation": "invalid",
                    "arm": "baseline",
                    "seed": 3234,
                    "points": [[0.0, 0.0]],
                },
            ).status_code
            == 422
        )
    return {
        "passed": True,
        "scope": "in-process API/agent safety and schema regression; not deployment or CPU model certification",
    }


def us_prediction_regression(predictor, rows):
    checks = []
    domains = {
        "wave2d": [[-1, 1], [0, 2], [0, 1]],
        "heat2d": [[0, 1]] * 3,
        "poisson2d": [[0, 1]] * 2,
        "burgers": [[-1, 1], [0, 1]],
        "kovasznay_forward": [[-0.5, 1], [-0.5, 1.5]],
        "kovasznay_inverse": [[-0.5, 1], [-0.5, 1.5]],
        "taylor_green": [[-np.pi, np.pi], [-np.pi, np.pi], [0, 1]],
        "darcy2d": [[0, 1]] * 2,
        "reaction_diffusion": [[0, 1]] * 3,
    }
    for r in rows:
        identity = {k: r[k] for k in ["equation", "arm", "seed"]}
        d = np.asarray(domains[r["equation"]], float)
        x = (d[:, 0] + np.array([[0.25], [0.65]]) * (d[:, 1] - d[:, 0])).tolist()
        try:
            a = np.asarray(predictor(**identity, points=x)["prediction"])
            b = np.asarray(predictor(**identity, points=x)["prediction"])
            singles = np.concatenate(
                [predictor(**identity, points=[v])["prediction"] for v in x]
            )
            if not np.array_equal(a, b) or not np.allclose(
                a, singles, rtol=1e-4, atol=1e-6
            ):
                raise ValueError("Repeated or single/batch inference mismatch")
            checks.append(
                {**identity, "status": "passed", "finite_repeat_single_batch": True}
            )
        except FileNotFoundError as exc:
            checks.append(
                {
                    **identity,
                    "status": "not_run_missing_model_artifact",
                    "reason": str(exc),
                }
            )
    return {
        "checks": checks,
        "passed": bool(checks) and all(c["status"] == "passed" for c in checks),
        "scope": "local inference regression only; no optimizer continuation or industrial certification",
    }
