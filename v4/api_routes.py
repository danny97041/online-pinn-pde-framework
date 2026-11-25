"""HTTP adapters. Training is not exposed; mutation routes default to disabled."""

from fastapi import FastAPI, HTTPException
from .api_schemas import Prediction, Question, Registration, Tool
from .reporting import comparison


def build_app(service, allow_registry_write=False):
    app = FastAPI(title="Online PINN PDE Framework V4")

    def safe(call):
        try:
            return call()
        except (ValueError, PermissionError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    app.get("/health")(
        lambda: {"status": "ok", "version": "v4", "training_endpoint": False}
    )
    app.get("/models")(service.models)
    app.get("/comparison")(lambda: comparison(service.root))
    app.get("/model/info")(lambda trial_id: safe(lambda: service.model_info(trial_id)))

    @app.post("/predict")
    @app.post("/predict/batch")
    def predict(request: Prediction):
        return safe(lambda: service.predict(request.trial_id, request.coordinates))

    @app.post("/agent/ask")
    def ask(request: Question):
        return safe(lambda: service.ask(**request.model_dump()))

    @app.post("/agent/tool")
    def tool(request: Tool):
        if (
            request.name in {"generate_report", "generate_feature_catalog"}
            and not allow_registry_write
        ):
            raise HTTPException(403, "Document writes disabled")
        return safe(lambda: service.tool(**request.model_dump()))

    @app.get("/rag/search")
    @app.get("/rag/context")
    def search(query: str, dense: bool = False, backend: str = "numpy"):
        return safe(lambda: service.search(query, dense, backend))

    @app.get("/report")
    def report():
        p = service.root / "REPORT.md"
        return {"text": p.read_text(encoding="utf-8") if p.exists() else ""}

    @app.post("/registry/select")
    def select(request: Registration):
        if not allow_registry_write:
            raise HTTPException(403, "Registry writes disabled")
        return safe(
            lambda: service.register(
                request.equation, request.trial_id, request.approved
            )
        )

    @app.post("/registry/rollback")
    def rollback(equation: str, approved: bool = False):
        if not allow_registry_write:
            raise HTTPException(403, "Registry writes disabled")
        return safe(lambda: service.rollback(equation, approved))

    @app.post("/predict/active")
    def active(equation: str, coordinates: list[list[float]]):
        trial = service.registry()["active"].get(equation)
        if trial is None:
            raise HTTPException(404, "No active model")
        return safe(lambda: service.predict(trial, coordinates))

    return app
