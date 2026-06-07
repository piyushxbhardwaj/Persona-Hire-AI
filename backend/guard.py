import re
import logging

logger = logging.getLogger("Guardrails")

# Common prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore\s+(?:previous|all|the|all\s+previous|your)?\s*instructions",
    r"reveal\s+.*prompt",
    r"output\s+.*prompt",
    r"print\s+.*prompt",
    r"you\s+are\s+now\s+a",
    r"pretend\s+you\s+are",
    r"forget\s+your\s+(?:grounding|instructions|persona)",
    r"bypass\s+restrictions",
    r"jailbreak",
    r"system\s+override",
    r"developer\s+mode",
    r"dan\s+mode",
]

# Compile patterns for efficiency
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

class InputGuard:
    @staticmethod
    def validate_query(query: str) -> tuple[bool, str]:
        """Validates a user query against prompt injection and security rules.
        
        Returns:
            (is_safe, refusal_message)
        """
        if not query or not query.strip():
            return False, "Query is empty."

        # Check for injection patterns
        for pattern in COMPILED_PATTERNS:
            if pattern.search(query):
                logger.warning(f"Guardrail triggered! Query matched pattern: {pattern.pattern}")
                # For sensitive credential requests, return grounding refusal as expected
                query_lower = query.lower()
                if any(kw in query_lower for kw in ["password", "token", "secret", "credential", "login", "key"]):
                    return False, "I don't know based on the available resume and GitHub data."
                return False, "I cannot fulfill this request. I am only programmed to discuss Piyush Bhardwaj's background, experience, projects, skills, or assist in scheduling interviews."

        # Length safety
        if len(query) > 1000:
            return False, "Query exceeds safe length threshold."

        return True, ""
