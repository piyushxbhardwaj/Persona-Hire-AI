import os
import json
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Header, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, EmailStr
from dotenv import load_dotenv

# Import components
from backend.rag_service import RAGService
from backend.calendar_service import CalendarService
from backend.memory import memory_manager
from backend.audit_logger import audit_logger

# Load env variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("FastAPIServer")

app = FastAPI(
    title="PersonaHire AI Backend API",
    description="Production API for candidate representation and calendar scheduling.",
    version="1.0.0"
)

# CORS configurations for Streamlit Cloud and browser accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
rag_service = RAGService()
calendar_service = CalendarService()

# ----------------------------------------------------
# Pydantic Schemas
# ----------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., example="What projects did Piyush build?")
    session_id: str = Field(default="default_session", example="session_123")

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str

class SlotRequest(BaseModel):
    date: str = Field(..., description="Format: YYYY-MM-DD", example="2026-06-05")

class BookRequest(BaseModel):
    date: str = Field(..., example="2026-06-05")
    slot: str = Field(..., example="10:00-11:00")
    email: EmailStr = Field(..., example="recruiter@scaler.com")
    name: str = Field(..., example="Anjali Sharma")

class CancelRequest(BaseModel):
    date: str = Field(..., example="2026-06-05")
    slot: str = Field(..., example="10:00-11:00")
    event_id: str = Field(..., example="evt_1234567")

# ----------------------------------------------------
# Health Check Endpoints
# ----------------------------------------------------
@app.get("/health", tags=["Health"])
def health_check():
    """Verify backend is alive."""
    return {"status": "healthy", "environment": os.getenv("ENV", "production")}

@app.get("/health/rag", tags=["Health"])
def health_rag():
    """Verify RAG vector store collections are loaded."""
    try:
        resume_count = rag_service.store.resume_col.count()
        github_count = rag_service.store.github_col.count()
        commit_count = rag_service.store.commit_col.count()
        
        return {
            "status": "healthy",
            "collections": {
                "resume_collection_size": resume_count,
                "github_collection_size": github_count,
                "commit_collection_size": commit_count
            }
        }
    except Exception as e:
        logger.error(f"RAG Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG Database offline: {e}"
        )

@app.get("/health/calendar", tags=["Health"])
def health_calendar():
    """Verify Calendar service status (Mock or Real)."""
    return {
        "status": "healthy",
        "calendar_mode": "Mock File Database" if calendar_service.use_mock else "Google Calendar API",
        "calendar_id": calendar_service.calendar_id
    }

# ----------------------------------------------------
# Chat & QA Endpoints
# ----------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
def chat_endpoint(request: ChatRequest):
    """Processes queries, retrieves context chunks, and synthesizes grounded responses."""
    try:
        result = rag_service.answer_query(request.query, request.session_id)
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=request.session_id
        )
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------------------------
# Calendar Booking Endpoints
# ----------------------------------------------------
@app.get("/api/calendar/slots", tags=["Calendar"])
def get_slots(date: str):
    """Get list of available booking slots for a date YYYY-MM-DD."""
    slots = calendar_service.get_available_slots(date)
    return {"date": date, "slots": slots}

@app.post("/api/calendar/book", tags=["Calendar"])
def book_appointment(request: BookRequest):
    """Books an interview slot with double-booking race condition protections."""
    success, event = calendar_service.create_event(
        date_str=request.date,
        slot=request.slot,
        attendee_email=request.email,
        attendee_name=request.name
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=event.get("error", "Slot is already booked.")
        )
        
    return {"success": True, "event": event}

@app.post("/api/calendar/cancel", tags=["Calendar"])
def cancel_appointment(request: CancelRequest):
    """Cancels an interview booking."""
    success = calendar_service.cancel_event(
        date_str=request.date,
        slot=request.slot,
        event_id=request.event_id
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not cancel appointment. Verify event ID, date, and slot."
        )
        
    return {"success": True}

# ----------------------------------------------------
# Vapi Voice Agent Integration Endpoints
# ----------------------------------------------------
@app.post("/api/voice/vapi", tags=["Voice"])
@app.post("/api/voice/vapi/chat/completions", tags=["Voice"])
@app.post("/api/voice/chat/completions", tags=["Voice"])
async def vapi_custom_llm(request: Request, x_vapi_secret: Optional[str] = Header(None, alias="x-vapi-secret")):
    """Handles Vapi voice rep integration.
    
    Acts as a Custom LLM endpoint for Vapi. Supports streaming answers from our grounded RAG.
    """
    # Optional webhook verification
    configured_secret = os.getenv("VAPI_SECRET")
    if configured_secret and x_vapi_secret != configured_secret:
        raise HTTPException(status_code=401, detail="Unauthorized voice request.")

    payload = await request.json()
    logger.info(f"Received Vapi custom LLM webhook call: {json.dumps(payload)[:300]}...")
    
    messages = payload.get("messages", [])
    if not messages:
        return {"choices": [{"message": {"role": "assistant", "content": "Hello! I am Piyush's AI Representative. How can I help you today?"}}]}
        
    # Get the last user message
    last_user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break
            
    if not last_user_msg:
        last_user_msg = "Hello"
        
    session_id = payload.get("call", {}).get("id", "vapi_session")
    
    # We query RAG
    # We request the RAG service to return the answer
    result = rag_service.answer_query(last_user_msg, session_id)
    content = result["answer"]
    
    # If the user requests streaming
    stream = payload.get("stream", False)
    
    if stream:
        def sse_generator():
            # Standard SSE stream formatting for Vapi
            # Vapi expects a stream matching OpenAI chunk responses
            chunk_size = 15
            words = content.split(" ")
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i+chunk_size]) + " "
                yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"
            yield "data: [DONE]\n\n"
            
        return StreamingResponse(sse_generator(), media_type="text/event-stream")
    else:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content
                    }
                }
            ]
        }

@app.post("/api/voice/vapi/webhook", tags=["Voice"])
async def vapi_tool_webhook(request: Request):
    """Processes Vapi tool calling requests.
    
    Maps tool calls (like calendar checking or booking) to our local services.
    """
    payload = await request.json()
    logger.info(f"Received Vapi tool call request: {json.dumps(payload)[:300]}")
    
    message = payload.get("message", {})
    msg_type = message.get("type")
    
    if msg_type == "tool-calls":
        tool_calls = message.get("toolCalls", [])
        results = []
        
        for call in tool_calls:
            call_id = call.get("id")
            name = call.get("function", {}).get("name")
            args = call.get("function", {}).get("arguments", {})
            
            logger.info(f"Vapi requesting execution of tool: {name} with args: {args}")
            
            result_str = ""
            success = True
            
            if name == "get_available_slots":
                date_str = args.get("date")
                slots = calendar_service.get_available_slots(date_str)
                result_str = f"Available slots for {date_str}: {', '.join(slots) if slots else 'No slots available.'}"
                
            elif name == "book_interview":
                date_str = args.get("date")
                slot = args.get("slot")
                email = args.get("email")
                name_attendee = args.get("name")
                
                ok, event_details = calendar_service.create_event(date_str, slot, email, name_attendee)
                if ok:
                    result_str = f"Interview successfully booked! Event ID: {event_details['event_id']}. Time: {slot} on {date_str}."
                else:
                    result_str = f"Failed to book interview: {event_details.get('error', 'Slot is taken.')}"
                    success = False
                    
            elif name == "cancel_interview":
                date_str = args.get("date")
                slot = args.get("slot")
                event_id = args.get("event_id")
                
                ok = calendar_service.cancel_event(date_str, slot, event_id)
                if ok:
                    result_str = f"Interview event {event_id} successfully cancelled."
                else:
                    result_str = f"Failed to cancel event {event_id}. Please verify event ID and time."
                    success = False
            else:
                result_str = f"Unknown function: {name}"
                success = False
                
            # Log to audit trail
            audit_logger.log_interaction(
                session_id=payload.get("call", {}).get("id", "vapi_webhook_session"),
                query=f"Voice Tool Call: {name}",
                retrieved_sources=[],
                tool_called=name,
                tool_args=args,
                tool_result=result_str,
                success=success
            )
            
            results.append({
                "toolCallId": call_id,
                "result": result_str
            })
            
        return {"results": results}
        
    return {"status": "ignored", "message": "unsupported message type"}

if __name__ == "__main__":
    import uvicorn
    # Use environment port or default to 8000
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
