import logging
import threading
from typing import List, Dict

logger = logging.getLogger("SessionMemory")

class SessionMemory:
    def __init__(self, max_history_len: int = 10):
        self.max_history_len = max_history_len
        # Maps session_id -> list of message dicts: [{"role": "user", "content": "..."}]
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
        self.lock = threading.Lock()

    def add_message(self, session_id: str, role: str, content: str):
        """Adds a message to the specified session thread. Limits size to max_history_len."""
        if not session_id:
            return
            
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = []
                
            self.sessions[session_id].append({"role": role, "content": content})
            
            # Keep only the last N messages
            if len(self.sessions[session_id]) > self.max_history_len:
                self.sessions[session_id] = self.sessions[session_id][-self.max_history_len:]
                
            logger.debug(f"Added message for session {session_id}. History size: {len(self.sessions[session_id])}")

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Returns the conversation history for a session."""
        if not session_id:
            return []
            
        with self.lock:
            return list(self.sessions.get(session_id, []))

    def clear_history(self, session_id: str):
        """Clears memory for a specific session."""
        if not session_id:
            return
            
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Cleared session history for: {session_id}")

    def format_history_for_llm(self, session_id: str) -> str:
        """Formats the session history into a single string for prompt injection."""
        history = self.get_history(session_id)
        formatted = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)

# Global singleton memory manager
memory_manager = SessionMemory()
