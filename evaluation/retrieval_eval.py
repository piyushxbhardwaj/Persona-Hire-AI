import os
import sys
import json
import logging
from typing import Dict, Any, List

# Ensure project root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from backend.rag_service import RAGService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RetrievalEvaluation")

def run_retrieval_evaluation():
    logger.info("Initializing RAG Service for retrieval evaluation...")
    rag = RAGService()
    
    # Load golden questions
    golden_path = os.path.join(current_dir, "golden_questions.json")
    if not os.path.exists(golden_path):
        logger.error(f"Golden questions file not found at: {golden_path}")
        return
        
    with open(golden_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
        
    results = []
    
    total_mrr = 0.0
    total_precision = 0.0
    total_recall = 0.0
    count_evaluable = 0
    
    for item in questions:
        q_id = item["id"]
        query = item["question"]
        category = item["category"]
        expected_sources = item["expected_sources"]
        
        # We only evaluate retrieval for queries expecting context (resume, github, commit)
        if not expected_sources:
            continue
            
        count_evaluable += 1
        logger.info(f"Evaluating retrieval for [{category}] Q: '{query}'")
        
        # Retrieve context
        top_chunks = rag.retrieve_context(query)
        
        # Calculate hits
        hits = []
        reciprocal_rank = 0.0
        found_first = False
        
        # Expected sources format in golden is: ["resume_collection", "github_collection", "commit_collection"]
        # Source collection retrieved is accessible from metadata in document details
        # Let's map metadata "source" to equivalent collections
        # resume.pdf -> resume_collection
        # GitHub (* README) / GitHub (*) -> github_collection
        # GitHub (* Commit History) -> commit_collection
        
        retrieved_sources_mapped = []
        for idx, chunk in enumerate(top_chunks[:5]):
            meta = chunk.get("metadata", {})
            source_name = meta.get("source", "").lower()
            
            mapped_col = ""
            if "resume" in source_name:
                mapped_col = "resume_collection"
            elif "commit" in source_name:
                mapped_col = "commit_collection"
            elif "github" in source_name:
                mapped_col = "github_collection"
                
            retrieved_sources_mapped.append(mapped_col)
            
            # Check relevance
            is_relevant = mapped_col in expected_sources
            hits.append(is_relevant)
            
            if is_relevant and not found_first:
                reciprocal_rank = 1.0 / (idx + 1)
                found_first = True
                
        # Precision@5: hits / retrieved (up to 5)
        p_at_5 = sum(hits) / len(top_chunks) if top_chunks else 0.0
        
        # Recall@5: unique relevant sources retrieved / total expected sources
        unique_retrieved_hits = set([src for src in retrieved_sources_mapped if src in expected_sources])
        r_at_5 = len(unique_retrieved_hits) / len(expected_sources) if expected_sources else 0.0
        
        total_mrr += reciprocal_rank
        total_precision += p_at_5
        total_recall += r_at_5
        
        results.append({
            "id": q_id,
            "query": query,
            "category": category,
            "expected_sources": expected_sources,
            "retrieved_sources": retrieved_sources_mapped,
            "precision_at_5": p_at_5,
            "recall_at_5": r_at_5,
            "reciprocal_rank": reciprocal_rank
        })
        
    avg_precision = total_precision / count_evaluable if count_evaluable else 0.0
    avg_recall = total_recall / count_evaluable if count_evaluable else 0.0
    avg_mrr = total_mrr / count_evaluable if count_evaluable else 0.0
    
    summary = {
        "total_evaluated_queries": count_evaluable,
        "average_precision_at_5": round(avg_precision * 100, 2),
        "average_recall_at_5": round(avg_recall * 100, 2),
        "average_mrr": round(avg_mrr, 4),
        "details": results
    }
    
    # Save output results
    output_path = os.path.join(parent_dir, "data", "retrieval_eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    logger.info(f"Retrieval evaluation complete. Saved results to {output_path}")
    logger.info(f"Precision@5: {summary['average_precision_at_5']}% | Recall@5: {summary['average_recall_at_5']}% | MRR: {summary['average_mrr']}")
    return summary

if __name__ == "__main__":
    run_retrieval_evaluation()
