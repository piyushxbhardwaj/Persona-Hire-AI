import os
import json
import logging
import base64
import requests
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GitHubLoader")

class GitHubLoader:
    def __init__(self, token: str = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.headers = {}
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"
        self.headers["Accept"] = "application/vnd.github.v3+json"
        self.base_url = "https://api.github.com"
        
        # Path to offline fallback dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.fallback_path = os.path.join(os.path.dirname(current_dir), "data", "github_mock.json")

    def _get_request(self, url: str) -> Any:
        response = requests.get(url, headers=self.headers, timeout=10)
        if response.status_code == 403 and "rate limit" in response.text.lower():
            logger.warning("GitHub API rate limit exceeded.")
            raise Exception("GitHub API rate limit exceeded.")
        response.raise_for_status()
        return response.json()

    def fetch_user_repos(self, username: str) -> List[Dict[str, Any]]:
        """Fetch all public repositories of a user."""
        url = f"{self.base_url}/users/{username}/repos?per_page=100&type=owner"
        return self._get_request(url)

    def fetch_repo_readme(self, username: str, repo_name: str) -> str:
        """Fetch and decode README for a repository."""
        try:
            url = f"{self.base_url}/repos/{username}/{repo_name}/readme"
            data = self._get_request(url)
            content = data.get("content", "")
            encoding = data.get("encoding", "")
            if encoding == "base64":
                return base64.b64decode(content).decode("utf-8", errors="ignore")
            return content
        except Exception as e:
            logger.warning(f"Could not fetch README for {repo_name}: {e}")
            return ""

    def fetch_repo_commits(self, username: str, repo_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch recent commits of a repository."""
        try:
            url = f"{self.base_url}/repos/{username}/{repo_name}/commits?per_page={limit}"
            commits_data = self._get_request(url)
            commits = []
            for item in commits_data:
                commit_info = item.get("commit", {})
                author_info = commit_info.get("author", {})
                commits.append({
                    "hash": item.get("sha", ""),
                    "message": commit_info.get("message", ""),
                    "date": author_info.get("date", ""),
                    "author": author_info.get("name", "")
                })
            return commits
        except Exception as e:
            logger.warning(f"Could not fetch commits for {repo_name}: {e}")
            return []

    def load_repositories(self, username: str) -> List[Dict[str, Any]]:
        """Loads repository metadata and commits. Falls back to mock data if offline or fails."""
        if not self.token:
            logger.info("No GITHUB_TOKEN found. Loading offline fallback data.")
            return self._load_fallback()

        try:
            logger.info(f"Connecting to GitHub API to ingest repos for user: {username}")
            repos = self.fetch_user_repos(username)
            results = []
            
            for repo in repos:
                repo_name = repo.get("name")
                logger.info(f"Ingesting details for repository: {repo_name}")
                readme = self.fetch_repo_readme(username, repo_name)
                commits = self.fetch_repo_commits(username, repo_name)
                
                results.append({
                    "name": repo_name,
                    "description": repo.get("description") or "",
                    "readme": readme,
                    "languages": [repo.get("language")] if repo.get("language") else [],
                    "topics": repo.get("topics", []),
                    "commits": commits
                })
            
            return results
        except Exception as e:
            logger.error(f"Failed to fetch from GitHub API ({e}). Falling back to offline dataset.")
            return self._load_fallback()

    def _load_fallback(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.fallback_path):
            logger.error(f"Fallback file not found at {self.fallback_path}!")
            return []
        try:
            with open(self.fallback_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} repositories from offline fallback JSON.")
                return data
        except Exception as e:
            logger.error(f"Error loading offline fallback JSON: {e}")
            return []

if __name__ == "__main__":
    # Quick test
    loader = GitHubLoader()
    repos_data = loader.load_repositories("piyushxbhardwaj")
    print(f"Loaded {len(repos_data)} repos.")
