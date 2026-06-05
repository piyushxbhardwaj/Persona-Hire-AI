import logging
from typing import List, Dict, Any

logger = logging.getLogger("Reranker")

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self.is_active = False
        
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading CrossEncoder reranker model: {self.model_name}...")
            # We set max_length to 512 for optimization
            self.model = CrossEncoder(self.model_name, max_length=512)
            self.is_active = True
            logger.info("CrossEncoder reranker loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load sentence-transformers CrossEncoder ({e}). "
                "Reranker will run in graceful fallback mode (using RRF scores directly)."
            )

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """Reranks candidates based on query similarity. Falls back to RRF rankings if CrossEncoder is offline."""
        if not candidates:
            return []
            
        if not self.is_active or not self.model:
            logger.debug("Reranker running in fallback mode (no-op, returning RRF sorted list).")
            # Items are already sorted by RRF score from ChromaStore
            return candidates[:top_n]
            
        try:
            logger.info(f"Reranking {len(candidates)} documents for query: '{query}'...")
            
            # Prepare query-document pairs
            pairs = [[query, doc["document"]] for doc in candidates]
            
            # Compute cross-encoder similarity scores
            scores = self.model.predict(pairs)
            
            # Append scores to candidates
            for idx, score in enumerate(scores):
                candidates[idx]["cross_encoder_score"] = float(score)
                
            # Re-sort candidates by cross-encoder score descending
            candidates.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
            
            logger.info(f"Reranking complete. Selected top {min(top_n, len(candidates))} documents.")
            return candidates[:top_n]
        except Exception as e:
            logger.error(f"Reranking failed: {e}. Falling back to default RRF rankings.")
            return candidates[:top_n]

if __name__ == "__main__":
    # Test class
    reranker = CrossEncoderReranker()
    test_docs = [
        {"id": "doc1", "document": "Piyush Bhardwaj is an AI Engineer from Chitkara University.", "metadata": {}},
        {"id": "doc2", "document": "Google Calendar API can schedule interviews.", "metadata": {}}
    ]
    results = reranker.rerank("Who is Piyush?", test_docs, top_n=1)
    print("Top reranked doc ID:", results[0]["id"] if results else "None")
