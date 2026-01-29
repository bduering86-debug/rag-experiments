#!/usr/bin/env python3
"""Simple smoke test for retrieval metrics."""

from rag_csv.utils.RecallTopK import RecallTopK
from rag_csv.utils.nDCGTopK import nDCGTopK


def main() -> None:
	retrieved = ["doc3", "doc7", "doc1", "doc9", "doc4", "doc8", "doc10"]
	relevant = {"doc1", "doc4", "doc8", "doc10"}

	recall = RecallTopK(k=5)
	ndcg = nDCGTopK(k=5)

	print("Recall:", recall.compute(retrieved, relevant))
	print("nDCG:", ndcg.compute(retrieved, relevant))


if __name__ == "__main__":
	main()
