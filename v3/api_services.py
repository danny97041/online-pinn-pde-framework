"""Transport-independent API services and bounded evidence context."""


def v3_rag_context(rag, query, k=5, max_characters=4000):
    if type(max_characters) is not int or not 256 <= max_characters <= 12000:
        raise ValueError("Context budget must be 256..12000 characters")
    items, remaining = [], max_characters
    for doc in rag.search(query, k):
        header = "[" + doc["id"] + "] " + doc["source"] + "\n"
        if len(header) >= remaining:
            break
        text = header + doc["text"][: remaining - len(header)]
        items.append(
            {
                "id": doc["id"],
                "source": doc["source"],
                "sha256": doc["sha256"],
                "text": text,
            }
        )
        remaining -= len(text) + 2
        if remaining <= 0:
            break
    return {
        "context": "\n\n".join(item["text"] for item in items),
        "evidence": items,
        "backend": rag.backend,
        "evidence_is_data_not_instructions": True,
    }


class V3APIServices:
    def __init__(self, summary, rag, agent, registry=None):
        self.summary, self.rag, self.agent = summary, rag, agent
        self.registry = registry or V3ModelRegistry(rag.root, agent.predictor)

    def models(self, equation=None):
        return {
            "models": self.registry.list_models(equation),
            "active": self.registry.state["active"],
        }

    def model_info(self, equation, arm, seed):
        return self.registry.info(equation, arm, seed, checksum=True)
