import os
import logging
from typing import List, Dict, Any, Tuple
import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

# Load env variables
load_dotenv()

logger = logging.getLogger("ChromaStore")

class ChromaStore:
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default to data/chroma_db in the project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(os.path.dirname(current_dir), "data", "chroma_db")
        
        logger.info(f"Initializing ChromaDB persistent client at: {db_path}")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # Initialize collections
        self.resume_col = self.chroma_client.get_or_create_collection(name="resume_collection")
        self.github_col = self.chroma_client.get_or_create_collection(name="github_collection")
        self.commit_col = self.chroma_client.get_or_create_collection(name="commit_collection")
        
        # Initialize OpenAI Client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY is not set in environment variables.")
        self.openai_client = OpenAI(api_key=api_key)

    def _get_embedding(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using text-embedding-3-small."""
        if not texts:
            return []
        try:
            # Check for dummy or missing key
            if not self.openai_client.api_key or "your_" in str(self.openai_client.api_key):
                raise ValueError("API Key is unconfigured or dummy.")
            
            response = self.openai_client.embeddings.create(
                input=[t.replace("\n", " ") for t in texts],
                model="text-embedding-3-small"
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.warning(f"Using offline deterministic mock embeddings (1536-dim) due to: {e}")
            import random
            mock_embeddings = []
            for t in texts:
                # Use string character sums as seed to keep embeddings deterministic for identical inputs
                seed = sum(ord(c) for c in t)
                rng = random.Random(seed)
                mock_embeddings.append([rng.uniform(-1, 1) for _ in range(1536)])
            return mock_embeddings

    def add_documents(self, collection_name: str, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """Embeds and adds documents to a specific collection."""
        if not documents:
            return
        
        # Select collection
        if collection_name == "resume":
            col = self.resume_col
        elif collection_name == "github":
            col = self.github_col
        elif collection_name == "commit":
            col = self.commit_col
        else:
            raise ValueError(f"Unknown collection: {collection_name}")
        
        logger.info(f"Generating embeddings for {len(documents)} docs in collection '{collection_name}'...")
        embeddings = self._get_embedding(documents)
        
        # Batch additions to Chroma (max 100 at a time)
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            col.add(
                documents=documents[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
        logger.info(f"Successfully added documents to '{collection_name}'.")

    def reset_collection(self, collection_name: str):
        """Clears all entries in a specific collection."""
        try:
            self.chroma_client.delete_collection(name=f"{collection_name}_collection")
            if collection_name == "resume":
                self.resume_col = self.chroma_client.get_or_create_collection(name="resume_collection")
            elif collection_name == "github":
                self.github_col = self.chroma_client.get_or_create_collection(name="github_collection")
            elif collection_name == "commit":
                self.commit_col = self.chroma_client.get_or_create_collection(name="commit_collection")
            logger.info(f"Reset collection '{collection_name}' successfully.")
        except Exception as e:
            logger.warning(f"Collection '{collection_name}' did not exist or failed to reset: {e}")

    def semantic_search(self, collection_name: str, query_text: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic vector search using text-embedding-3-small."""
        if collection_name == "resume":
            col = self.resume_col
        elif collection_name == "github":
            col = self.github_col
        elif collection_name == "commit":
            col = self.commit_col
        else:
            raise ValueError(f"Unknown collection: {collection_name}")

        try:
            query_emb = self._get_embedding([query_text])[0]
            results = col.query(
                query_embeddings=[query_emb],
                n_results=n_results
            )
            
            # Format results
            formatted = []
            if results and results["documents"] and len(results["documents"][0]) > 0:
                for idx in range(len(results["documents"][0])):
                    formatted.append({
                        "id": results["ids"][0][idx],
                        "document": results["documents"][0][idx],
                        "metadata": results["metadatas"][0][idx],
                        "distance": results["distances"][0][idx] if "distances" in results and results["distances"] else 0.0
                    })
            return formatted
        except Exception as e:
            logger.error(f"Semantic search failed for '{collection_name}': {e}")
            return []

    def bm25_search(self, collection_name: str, query_text: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Perform keyword search using BM25Okapi over local collection content."""
        if collection_name == "resume":
            col = self.resume_col
        elif collection_name == "github":
            col = self.github_col
        elif collection_name == "commit":
            col = self.commit_col
        else:
            raise ValueError(f"Unknown collection: {collection_name}")

        try:
            all_docs = col.get()
            if not all_docs or not all_docs["documents"]:
                return []
            
            documents = all_docs["documents"]
            metadatas = all_docs["metadatas"]
            ids = all_docs["ids"]
            
            # Tokenize corpus and query
            tokenized_corpus = [doc.lower().split() for doc in documents]
            tokenized_query = query_text.lower().split()
            
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(tokenized_query)
            
            # Combine documents with scores
            combined = []
            for idx, score in enumerate(scores):
                if score > 0:  # Only keep documents with keyword overlap
                    combined.append({
                        "id": ids[idx],
                        "document": documents[idx],
                        "metadata": metadatas[idx],
                        "bm25_score": score
                    })
            
            # Sort by score descending
            combined.sort(key=lambda x: x["bm25_score"], reverse=True)
            return combined[:n_results]
        except Exception as e:
            logger.error(f"BM25 search failed for '{collection_name}': {e}")
            return []

    def hybrid_search(self, collection_name: str, query_text: str, n_results: int = 10, k_rf: int = 60) -> List[Dict[str, Any]]:
        """Combines semantic search and BM25 search using Reciprocal Rank Fusion (RRF)."""
        # Fetch initial pool from both strategies (e.g., top 20 each)
        pool_size = max(20, n_results * 2)
        
        semantic_res = self.semantic_search(collection_name, query_text, n_results=pool_size)
        bm25_res = self.bm25_search(collection_name, query_text, n_results=pool_size)
        
        # Map item IDs to their data
        id_to_doc = {}
        for r in semantic_res:
            id_to_doc[r["id"]] = r
        for r in bm25_res:
            id_to_doc[r["id"]] = r
            
        # Perform Reciprocal Rank Fusion
        rrf_scores = {}
        
        # Process semantic ranks
        for rank, r in enumerate(semantic_res):
            doc_id = r["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_rf + (rank + 1)))
            
        # Process BM25 ranks
        for rank, r in enumerate(bm25_res):
            doc_id = r["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_rf + (rank + 1)))
            
        # Sort items by fused RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Construct final output list
        fused_results = []
        for doc_id in sorted_ids[:n_results]:
            doc_info = id_to_doc[doc_id].copy()
            doc_info["rrf_score"] = rrf_scores[doc_id]
            fused_results.append(doc_info)
            
        return fused_results

if __name__ == "__main__":
    # Test persistency and collection sizes
    store = ChromaStore()
    print("Resume Collection size:", store.resume_col.count())
    print("GitHub Collection size:", store.github_col.count())
    print("Commit Collection size:", store.commit_col.count())
