"""
Phase 4 evaluation using RAGAS against local Ollama.

Usage:
    pip install ragas
    python eval/ragas_eval.py

Requires a golden dataset: list of {question, ground_truth, contexts, answer} dicts.
"""
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall


GOLDEN_QA = [
    # Add your test cases here:
    # {
    #     "question": "What is the refund policy?",
    #     "ground_truth": "Refunds are processed within 30 days.",
    #     "contexts": ["...retrieved context chunk..."],
    #     "answer": "...LLM answer...",
    # },
]


def run_eval():
    if not GOLDEN_QA:
        print("No test cases defined. Add entries to GOLDEN_QA to run evaluation.")
        return

    dataset = Dataset.from_list(GOLDEN_QA)
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
    )
    print(results)


if __name__ == "__main__":
    run_eval()
