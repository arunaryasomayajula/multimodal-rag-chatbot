"""
LangGraph state graph for the RAG chatbot.

Flow:
  START → route_intent → [rag branch] retrieve → rerank → generate → END
                        → [tabular branch] tabular_predict → END
"""
import re
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank
from generation.llm_client import chat
from generation.tabular_predictor import predict_from_session


_TABULAR_PATTERN = re.compile(
    r"\b(predict|classify|forecast|regress|inference on|what will .+ be)\b",
    re.IGNORECASE,
)


class GraphState(TypedDict):
    # Inputs
    query: str
    user_id: str
    session_id: str
    model_id: str | None
    history: list[dict]
    active_dataset_id: str | None   # set when user has an active CSV in session

    # Routing
    route: str   # "rag" | "tabular"

    # RAG intermediate
    query_vector: list[float]
    candidates: list[dict]
    chunks: list[dict]

    # Output
    answer: str
    sources: list[dict]


def _route_intent(state: GraphState) -> dict:
    has_dataset = bool(state.get("active_dataset_id"))
    is_tabular_query = bool(_TABULAR_PATTERN.search(state["query"]))
    route = "tabular" if (has_dataset and is_tabular_query) else "rag"
    return {"route": route}


def _retrieve_node(state: GraphState) -> dict:
    from ingestion.embedder import embed_query
    qvec = embed_query(state["query"])
    candidates = hybrid_search(state["query"], qvec, user_id=state["user_id"])
    return {"query_vector": qvec, "candidates": candidates}


def _rerank_node(state: GraphState) -> dict:
    chunks = rerank(state["query"], state["candidates"])
    return {"chunks": chunks}


def _generate_node(state: GraphState) -> dict:
    if not state["chunks"]:
        return {
            "answer": "No relevant documents found. Please upload documents first.",
            "sources": [],
        }
    result = chat(state["query"], state["chunks"], state["history"], state.get("model_id"))
    return {"answer": result["answer"], "sources": result["sources"]}


def _tabular_predict_node(state: GraphState) -> dict:
    result = predict_from_session(
        dataset_id=state["active_dataset_id"],
        query=state["query"],
        user_id=state["user_id"],
    )
    return {"answer": result["summary"], "sources": [], "chunks": []}


def _route_branch(state: GraphState) -> str:
    return state["route"]


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("route_intent", _route_intent)
    g.add_node("retrieve", _retrieve_node)
    g.add_node("rerank", _rerank_node)
    g.add_node("generate", _generate_node)
    g.add_node("tabular_predict", _tabular_predict_node)

    g.add_edge(START, "route_intent")
    g.add_conditional_edges("route_intent", _route_branch, {
        "rag": "retrieve",
        "tabular": "tabular_predict",
    })
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "generate")
    g.add_edge("generate", END)
    g.add_edge("tabular_predict", END)

    return g.compile()


# Compiled once at import / app startup; reused per request.
rag_graph = build_graph()
