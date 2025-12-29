#!/usr/bin/env python3
"""Simple smoke test for retrieval metrics."""

try:
	# Prefer package-style import when run from project root or as a module
	from metrics.retrievalquaility.RecallTopK import RecallTopK
	from metrics.retrievalquaility.nDCGTopK import nDCGTopK
except (ModuleNotFoundError, ImportError):
	# Fallback for direct execution (`./test_rq.py`) where the script's
	# directory is on sys.path: import local modules by filename.
	from RecallTopK import RecallTopK
	from nDCGTopK import nDCGTopK


def main() -> None:
	retrieved = ["doc3", "doc7", "doc1", "doc9", "doc4", "doc8", "doc10"]
	relevant = {"doc1", "doc4", "doc8", "doc10"}

	recall = RecallTopK(k=5)
	ndcg = nDCGTopK(k=5)

	print("Recall:", recall.compute(retrieved, relevant))
	print("nDCG:", ndcg.compute(retrieved, relevant))


if __name__ == "__main__":
	main()
