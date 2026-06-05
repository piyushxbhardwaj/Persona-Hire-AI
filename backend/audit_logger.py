import os
import json
import datetime
import threading
import logging
from typing import List, Dict, Any

logger = logging.getLogger("AuditLogger")

class AuditLogger:
    def __init__(self, log_path: str = None):
        if log_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            log_path = os.path.join(os.path.dirname(current_dir), "data", "audit_log.jsonl")
            
        self.log_path = log_path
        self.lock = threading.Lock()
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log_interaction(
        self,
        session_id: str,
        query: str,
        retrieved_sources: List[str],
        tool_called: str = None,
        tool_args: Dict[str, Any] = None,
        tool_result: str = None,
        success: bool = True
    ):
        """Thread-safe append of a structured audit entry to data/audit_log.jsonl."""
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "session_id": session_id,
            "user_query": query,
            "retrieved_sources": retrieved_sources,
            "tool_called": tool_called,
            "tool_args": tool_args,
            "tool_result": tool_result,
            "success": success
        }
        
        try:
            with self.lock:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            logger.debug(f"Audit log appended successfully to {self.log_path}")
        except Exception as e:
            logger.error(f"Failed to write to audit log: {e}")

# Global singleton audit logger
audit_logger = AuditLogger()
