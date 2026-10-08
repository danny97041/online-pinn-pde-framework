"""FastAPI router composition. Write routes are disabled unless explicitly enabled."""


def v3_api_app(summary, rag, agent, registry_write=False, registry=None):
    from fastapi import FastAPI, APIRouter, HTTPException

    schemas = v3_api_schemas()
    (
        Search,
        Context,
        Command,
        Prediction,
        Question,
        Registration,
        Rollback,
        ActivePrediction,
    ) = (
        schemas[name]
        for name in [
            "Search",
            "Context",
            "Command",
            "Prediction",
            "Question",
            "Registration",
            "Rollback",
            "ActivePrediction",
        ]
    )
    service = V3APIServices(summary, rag, agent, registry)
    router = APIRouter()

    def invoke(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, TypeError, FileNotFoundError, ZeroDivisionError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/health")
    def health():
        return {"status": "ok", "research_only": True, "automatic_promotion": False}

    @router.get("/comparison")
    def comparison(equation: str = "wave2d"):
        return invoke(agent.execute, "compare", {"equation": equation})

    @router.get("/report")
    def report():
        return summary

    @router.get("/models")
    def models(equation: str | None = None):
        return invoke(service.models, equation)

    @router.get("/model/info")
    def model_info(equation: str, arm: str, seed: int):
        return invoke(service.model_info, equation, arm, seed)

    @router.post("/rag/search")
    def search(req: Search):
        return {"results": invoke(rag.search, req.query, req.k), "backend": rag.backend}

    @router.post("/rag/context")
    def context(req: Context):
        return invoke(v3_rag_context, rag, **req.model_dump())

    @router.post("/agent/tool")
    def tool(req: Command):
        return invoke(agent.execute, req.tool, req.arguments)

    @router.post("/agent/ask")
    def ask(req: Question):
        return invoke(agent.ask, req.question)

    @router.post("/predict")
    @router.post("/predict/batch")
    def predict(req: Prediction):
        return invoke(agent.execute, "predict", req.model_dump())

    @router.post("/predict/active")
    def predict_active(req: ActivePrediction):
        return invoke(service.registry.predict_active, **req.model_dump())

    @router.get("/registry")
    def registry_state():
        return service.registry.state

    @router.post("/registry/activate")
    def activate(req: Registration):
        if not registry_write:
            raise HTTPException(403, "Registry write routes are disabled")
        return invoke(service.registry.activate, **req.model_dump())

    @router.post("/registry/rollback")
    def rollback(req: Rollback):
        if not registry_write:
            raise HTTPException(403, "Registry write routes are disabled")
        return {"active": invoke(service.registry.rollback, **req.model_dump())}

    app = FastAPI(title="V3 PINN API", version="3.1-service")
    app.include_router(router)
    app.state.services = service
    return app
