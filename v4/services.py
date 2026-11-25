"""Read-only API, hybrid RAG, tool-routed questions and manual inference registry."""

import json
from pathlib import Path
import re
import numpy as np

from .config import Settings, ARM_LABELS
from .data import DataSource
from .equations import EQUATIONS
from .reporting import comparison, generate_report, reports
from .storage import write_json, sha_file


class Service:
    def __init__(self, root):
        self.root = Path(root)
        self.cache = {}
        self.registry_path = self.root / "registry.json"

    def models(self):
        return [
            {
                "trial_id": r["trial_id"],
                "equation": r["equation"],
                "inverse": r["inverse"],
                "arm": r["arm"],
                "seed": r["seed"],
                "selected_stage": r["selected_stage"],
                "outputs": EQUATIONS[r["equation"]]["outputs"],
                "generalization": r["generalization"],
                "checkpoint_exists": (
                    self.root / "trials" / r["trial_id"] / "selected.npz"
                ).exists(),
            }
            for r in reports(self.root)
        ]

    def model_info(self, trial_id):
        found = next((r for r in self.models() if r["trial_id"] == trial_id), None)
        if found is None:
            raise ValueError("완료 모델이 없습니다.")
        cfg = json.loads(
            (self.root / "trials" / trial_id / "trial_settings.json").read_text(
                encoding="utf-8"
            )
        )
        spec = EQUATIONS[found["equation"]]
        source = DataSource(
            Settings.from_dict(cfg["settings"]),
            (
                self.root / "inputs/data.json"
                if cfg["settings"]["data_mode"] == "csv"
                else None
            ),
        )
        return {
            **found,
            "input_variables": [
                {"symbol": a, "range": d} for a, d in zip(spec["axes"], source.domain)
            ],
            "output_variables": spec["outputs"],
            "coefficient": spec["coefficient"],
            "settings": cfg["settings"],
        }

    def predict(self, trial_id, coordinates):
        from .runtime import Runtime

        info = self.model_info(trial_id)
        points = np.asarray(coordinates, float)
        if (
            points.ndim != 2
            or not 1 <= len(points) <= 1024
            or points.shape[1] != len(info["input_variables"])
            or not np.isfinite(points).all()
        ):
            raise ValueError("예측 좌표 차원·크기를 확인하세요.")
        domain = np.array([v["range"] for v in info["input_variables"]])
        if np.any(points < domain[:, 0]) or np.any(points > domain[:, 1]):
            raise ValueError("저장 영역 밖의 좌표입니다.")
        if trial_id not in self.cache:
            settings = Settings.from_dict(info["settings"])
            source = DataSource(
                settings,
                self.root / "inputs/data.json" if settings.data_mode == "csv" else None,
            )
            rt = Runtime(settings, info["arm"], source)
            rt.load(self.root / "trials" / trial_id / "selected.npz")
            self.cache[trial_id] = rt
        return {
            "trial_id": trial_id,
            "prediction": self.cache[trial_id].predict(points).tolist(),
            "outputs": info["output_variables"],
        }

    def registry(self):
        return (
            json.loads(self.registry_path.read_text(encoding="utf-8"))
            if self.registry_path.exists()
            else {"active": {}, "history": []}
        )

    def register(self, equation, trial_id, approved=False):
        if approved is not True:
            raise PermissionError("모델 변경 승인이 필요합니다.")
        info = self.model_info(trial_id)
        if info["equation"] != equation:
            raise ValueError("방정식 불일치")
        if not info["checkpoint_exists"]:
            raise ValueError("완료 모델 파일이 없습니다.")
        center = [[sum(v["range"]) / 2 for v in info["input_variables"]]]
        prediction = self.predict(trial_id, center)
        if not np.isfinite(prediction["prediction"]).all():
            raise ValueError("추론 모델 사전 검사 실패")
        validation = {
            "checkpoint_sha256": sha_file(
                self.root / "trials" / trial_id / "selected.npz"
            ),
            "settings_sha256": sha_file(
                self.root / "trials" / trial_id / "trial_settings.json"
            ),
            "model_load_and_finite_prediction": True,
        }
        state = self.registry()
        previous = state["active"].get(equation)
        state["history"].append(
            {
                "equation": equation,
                "previous": previous,
                "selected": trial_id,
                "validation": validation,
            }
        )
        state["active"][equation] = trial_id
        write_json(self.registry_path, state)
        return state

    def rollback(self, equation, approved=False):
        if approved is not True:
            raise PermissionError("모델 변경 승인이 필요합니다.")
        state = self.registry()
        item = next(
            (v for v in reversed(state["history"]) if v["equation"] == equation), None
        )
        if item is None or item["previous"] is None:
            raise ValueError("복원할 직전 모델이 없습니다.")
        return self.register(equation, item["previous"], True)

    def search(self, query, dense=False, backend="numpy", count=5):
        if (
            backend not in {"numpy", "faiss"}
            or not isinstance(query, str)
            or not 1 <= len(query) <= 2000
            or type(count) is not int
            or not 1 <= count <= 20
        ):
            raise ValueError("검색 입력·개수·엔진을 확인하세요.")
        documents = []
        for p in [self.root / "REPORT.md", self.root / "FEATURES.md"]:
            if p.exists():
                text = p.read_text(encoding="utf-8")
                documents.extend(
                    {"source": p.name, "text": text[i : i + 800]}
                    for i in range(0, len(text), 600)
                )
        if not documents:
            return []
        tokenize = lambda s: re.findall(r"[\w]+", s.lower())
        words = [tokenize(d["text"]) for d in documents]
        terms = tokenize(query)
        n = len(words)
        length = np.array([len(v) for v in words])
        average = max(float(length.mean()), 1.0)
        scores = np.zeros(n)
        for term in terms:
            frequency = np.array([v.count(term) for v in words])
            df = sum(term in v for v in words)
            idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
            scores += (
                idf
                * frequency
                * 2.5
                / (frequency + 1.5 * (0.25 + 0.75 * length / average))
            )
        order = np.argsort(-scores)
        ranks = 0.4 / (60 + np.argsort(order) + 1)
        if dense:
            from sentence_transformers import SentenceTransformer

            if not hasattr(self, "embedding"):
                self.embedding = SentenceTransformer("intfloat/multilingual-e5-small")
            embeddings = np.asarray(
                self.embedding.encode(
                    ["passage: " + d["text"] for d in documents],
                    normalize_embeddings=True,
                ),
                np.float32,
            )
            q = np.asarray(
                self.embedding.encode(["query: " + query], normalize_embeddings=True),
                np.float32,
            )
            if backend == "faiss":
                import faiss

                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings)
                _, indices = index.search(q, n)
                dense_order = indices[0]
            elif backend == "numpy":
                dense_order = np.argsort(-(embeddings @ q[0]))
            else:
                raise ValueError("검색 엔진 오류")
            ranks += 0.6 / (60 + np.argsort(dense_order) + 1)
        return [
            {
                **documents[i],
                "score": float(ranks[i]),
                "backend": "BM25" + (" + E5/" + backend + " + RRF" if dense else ""),
            }
            for i in np.argsort(-ranks)[:count]
        ]

    def tool(self, name, arguments=None, approved=False):
        arguments = arguments or {}
        if name == "comparison":
            return comparison(self.root)
        if name == "models":
            return self.models()
        if name == "model_info":
            return self.model_info(arguments.get("trial_id", ""))
        if name == "predict":
            return self.predict(
                arguments.get("trial_id", ""), arguments.get("coordinates", [])
            )
        if name == "search":
            return self.search(**arguments)
        if name in {"generate_report", "generate_feature_catalog"}:
            if approved is not True:
                raise PermissionError("문서 생성 승인이 필요합니다.")
            generate_report(self.root)
            file = self.root / (
                "REPORT.md" if name == "generate_report" else "FEATURES.md"
            )
            return {
                "text": file.read_text(encoding="utf-8"),
                "rag_reindex": "next search reads current generated documents",
            }
        raise ValueError("허용되지 않은 도구입니다.")

    def ask(self, question, use_llm=False, dense=False, backend="numpy"):
        if not isinstance(question, str) or not 1 <= len(question) <= 2000:
            raise ValueError("질문 길이는 1~2000자입니다.")
        if re.search(
            r"삭제|덮어|비밀번호|토큰|delete|overwrite|학습.*실행", question, re.I
        ):
            return {
                "answer": "질문으로 파일 변경이나 학습을 실행하지 않습니다. 해당 실행 패널에서 선택하세요.",
                "tool": "refuse",
                "llm_called": False,
            }
        equations = [
            k
            for k, v in EQUATIONS.items()
            if k.lower() in question.lower() or v["label"] in question
        ]
        aliases = {
            "포아송": "poisson2d",
            "파동": "wave2d",
            "열전도": "heat2d",
            "버거스": "burgers",
            "Darcy": "darcy2d",
            "반응": "reaction_diffusion",
            "Helmholtz": "helmholtz",
        }
        equations += [v for k, v in aliases.items() if k.lower() in question.lower()]
        if len(set(equations)) > 1:
            return {
                "answer": "방정식을 하나 지정해 주세요. 방정식마다 조건을 구분해서 비교합니다.",
                "tool": "clarify",
                "llm_called": False,
            }
        equation = next(iter(dict.fromkeys(equations)), None)
        called = False
        if re.search(r"예측|predict", question, re.I):
            match = re.search(r"\[\s*\[.*?\]\s*\]", question)
            active = self.registry()["active"].get(equation)
            if match and active:
                prediction = self.predict(active, json.loads(match.group()))
                return {
                    "answer": "저장 모델의 예측값은 "
                    + json.dumps(prediction["prediction"], ensure_ascii=False)
                    + "입니다.",
                    "tool": "predict",
                    "evidence": prediction,
                    "llm_called": False,
                }
            return {
                "answer": "좌표 JSON과 방정식을 지정하고 기본 추론 모델을 선택해 주세요.",
                "tool": "clarify",
                "llm_called": False,
            }
        if use_llm:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            if not hasattr(self, "llm"):
                import torch

                self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
                self.llm = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen3.5-0.8B", dtype="auto"
                )
                self.llm.to("cuda" if torch.cuda.is_available() else "cpu")
                self.llm.eval()
            prompt = (
                'Return JSON only: {"equation": supported_name_or_null, "tool": "comparison" or "search"}. No commands. Supported: '
                + ",".join(EQUATIONS)
                + "\nQuestion: "
                + question
            )
            messages = [{"role": "user", "content": prompt}]
            encoded = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
                enable_thinking=False,
            ).to(self.llm.device)
            import torch

            with torch.inference_mode():
                generated = self.llm.generate(
                    **encoded, max_new_tokens=96, do_sample=False
                )
            text = self.tokenizer.decode(
                generated[0][encoded["input_ids"].shape[-1] :], skip_special_tokens=True
            )
            called = True
            try:
                proposed = json.loads(text[text.index("{") : text.rindex("}") + 1])
                if equation is None and proposed.get("equation") in EQUATIONS:
                    equation = proposed["equation"]
            except (ValueError, TypeError):
                pass
        rows = [
            r
            for r in comparison(self.root)
            if equation is None or r["equation"] == equation
        ]
        if equation and not rows:
            return {
                "answer": EQUATIONS[equation]["label"]
                + "의 완료된 V4 비교 실험이 아직 없습니다.",
                "tool": "comparison",
                "evidence": [],
                "llm_called": called,
            }
        if equation and rows:
            requested_stage = (
                "lbfgs" if re.search(r"L.?BFGS|후처리", question, re.I) else "adam"
            )
            if "역문제" in question:
                rows = [r for r in rows if r["inverse"]]
            elif "순방향" in question:
                rows = [r for r in rows if not r["inverse"]]
            rows = [
                r for r in rows if r["stage"] == requested_stage and r["l2"] is not None
            ]
            if rows:
                partitions = {
                    (
                        r["inverse"],
                        r["generalization"],
                        r["execution_scope"],
                        r["noise_fraction"],
                        r["initial_coefficient"],
                        r["protocol_sha"],
                    )
                    for r in rows
                }
                if len(partitions) > 1:
                    return {
                        "answer": "순·역문제와 기본·보간·외삽 실험을 구분해서 비교합니다. 원하는 범위를 지정해 주세요.",
                        "tool": "clarify",
                        "llm_called": called,
                    }
                best = min(rows, key=lambda r: r["l2"])
                answer = f"{EQUATIONS[equation]['label']}의 저장 {requested_stage} 결과에서 물리장 L2 오차가 가장 낮은 개별 실험은 {ARM_LABELS[best['arm']]} / 시드 {best['seed']}이며 {best['l2']*100:.5f}%입니다. 저장된 결과 비교입니다."
                return {
                    "answer": answer,
                    "tool": "comparison",
                    "evidence": rows,
                    "llm_called": called,
                }
            return {
                "answer": "요청한 방향·단계의 완료 결과가 없습니다.",
                "tool": "comparison",
                "evidence": [],
                "llm_called": called,
            }
        context = self.search(question, dense, backend)
        return {
            "answer": (
                "확인할 방정식·순/역문제·Adam/L-BFGS 단계를 지정해 주세요."
                if not rows
                else f"완료된 V4 실험 {len(reports(self.root))}개가 있습니다. 방정식을 지정하면 비교할 수 있습니다."
            ),
            "tool": "search",
            "context": context,
            "llm_called": called,
        }


def create_app(service, allow_registry_write=False):
    from .api_routes import build_app

    return build_app(service, allow_registry_write)
