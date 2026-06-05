import os
import sys
import logging
from pypdf import PdfReader
from dotenv import load_dotenv

# Ensure the root of the workspace is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from backend.github_loader import GitHubLoader
from backend.chroma_store import ChromaStore

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IngestionPipeline")

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list:
    """Chunks text with sliding window overlap."""
    chunks = []
    if not text:
        return chunks
    
    # Simple character-based sliding window
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += (chunk_size - overlap)
    
    return chunks

def ingest_resume(store: ChromaStore, pdf_path: str):
    """Parses resume PDF, chunks pages, and inserts into ChromaDB."""
    if not os.path.exists(pdf_path):
        logger.error(f"Resume PDF not found at: {pdf_path}")
        return
        
    logger.info(f"Opening resume PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    
    documents = []
    metadatas = []
    ids = []
    
    chunk_counter = 0
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
            
        logger.info(f"Processing page {page_num + 1}...")
        
        # Chunk page content
        page_chunks = chunk_text(text, chunk_size=700, overlap=120)
        for sub_idx, chunk in enumerate(page_chunks):
            doc_id = f"resume_p{page_num + 1}_c{sub_idx + 1}"
            documents.append(chunk)
            metadatas.append({
                "source": "resume.pdf",
                "page": page_num + 1,
                "section": "Resume General",
                "id": doc_id
            })
            ids.append(doc_id)
            chunk_counter += 1
            
    # Reset collection first
    store.reset_collection("resume")
    
    # Save documents
    store.add_documents("resume", documents, metadatas, ids)
    logger.info(f"Resume ingestion complete. Added {chunk_counter} chunks.")

def ingest_github(store: ChromaStore, username: str):
    """Loads repo and commit metadata and inserts into ChromaDB."""
    loader = GitHubLoader()
    repos = loader.load_repositories(username)
    
    if not repos:
        logger.warning("No GitHub repositories to ingest.")
        return
        
    # Reset collections
    store.reset_collection("github")
    store.reset_collection("commit")
    
    repo_documents = []
    repo_metadatas = []
    repo_ids = []
    
    commit_documents = []
    commit_metadatas = []
    commit_ids = []
    
    for repo in repos:
        repo_name = repo["name"]
        description = repo["description"]
        readme = repo["readme"]
        languages = ", ".join(repo["languages"])
        topics = ", ".join(repo["topics"])
        
        # 1. Ingest repository overview doc
        overview_text = f"Repository: {repo_name}\nDescription: {description}\nLanguages: {languages}\nTopics: {topics}"
        overview_id = f"github_repo_{repo_name}_overview"
        repo_documents.append(overview_text)
        repo_metadatas.append({
            "source": f"GitHub ({repo_name})",
            "repo_name": repo_name,
            "type": "overview",
            "id": overview_id
        })
        repo_ids.append(overview_id)
        
        # 2. Chunk and ingest README
        if readme:
            readme_chunks = chunk_text(readme, chunk_size=800, overlap=150)
            for idx, chunk in enumerate(readme_chunks):
                readme_id = f"github_repo_{repo_name}_readme_c{idx + 1}"
                repo_documents.append(f"Repository: {repo_name} (README)\n{chunk}")
                repo_metadatas.append({
                    "source": f"GitHub ({repo_name} README)",
                    "repo_name": repo_name,
                    "type": "readme",
                    "id": readme_id
                })
                repo_ids.append(readme_id)
                
        # 3. Ingest Commits
        commits = repo.get("commits", [])
        for commit in commits:
            commit_hash = commit["hash"]
            message = commit["message"]
            date = commit["date"]
            author = commit["author"]
            
            commit_text = (
                f"Repository: {repo_name}\n"
                f"Commit Hash: {commit_hash}\n"
                f"Author: {author}\n"
                f"Date: {date}\n"
                f"Message: {message}"
            )
            
            commit_id = f"github_commit_{repo_name}_{commit_hash}"
            commit_documents.append(commit_text)
            commit_metadatas.append({
                "source": f"GitHub ({repo_name} Commit History)",
                "repo_name": repo_name,
                "commit_hash": commit_hash,
                "author": author,
                "date": date,
                "type": "commit",
                "id": commit_id
            })
            commit_ids.append(commit_id)

    # Insert into collections
    logger.info(f"Ingesting {len(repo_documents)} repo overview and README chunks...")
    store.add_documents("github", repo_documents, repo_metadatas, repo_ids)
    
    logger.info(f"Ingesting {len(commit_documents)} commit log documents...")
    store.add_documents("commit", commit_documents, commit_metadatas, commit_ids)
    
    logger.info("GitHub and Commit ingestion complete.")

def run_pipeline():
    """Initializes and runs the full ingestion pipeline."""
    # Ensure raw data folder exists
    data_dir = os.path.join(parent_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    store = ChromaStore()
    
    # Path to resume
    pdf_path = os.path.join(data_dir, "resume.pdf")
    
    # Ingest resume
    ingest_resume(store, pdf_path)
    
    # Ingest github info
    ingest_github(store, "piyushxbhardwaj")
    
    logger.info("Pipeline executed successfully!")

if __name__ == "__main__":
    run_pipeline()
