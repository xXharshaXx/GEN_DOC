from fastapi import FastAPI
from pydantic import BaseModel
from .ai_agent import build_graph, ChatInfo, ProjectDetails, llm
from .ai_agent import DynamicExtractor
from .ai_agent import SuggestionAgent
from .ai_validator import AIValidator
from .chat_history import load_history, append_turn
from .prompts import PROMPTS
from .token_utils import TokenTracker, count_tokens_in_messages
import logging
import os
import re
import time


app = FastAPI(title="LangGraph Document Generation API")
graph = build_graph()
dynamic_extractor = DynamicExtractor()
ai_validator = AIValidator()
suggestion_agent = SuggestionAgent()


class RunResponse(BaseModel):
    reply: str


class UnifiedAnalysisResponse(BaseModel):
    classification: dict
    extracted_data: dict
    validation: dict
    key_points: list[str]
    suggestions: list[str]
    citations: list[dict]
    false_info_detected: bool
    success: bool
    message: str


def clean_response(content: str) -> str:
    """Remove thinking patterns and return only the final response"""
    # Remove common thinking patterns
    thinking_patterns = [
        r'<thinking>.*?</thinking>',
        r'<think>.*?</think>',
        r'Thinking:.*?(?=\n\n|\n[A-Z]|$)',
        r'Let me think.*?(?=\n\n|\n[A-Z]|$)',
        r'I need to.*?(?=\n\n|\n[A-Z]|$)',
        r'First, let me.*?(?=\n\n|\n[A-Z]|$)',
    ]
    
    
    cleaned = content
    for pattern in thinking_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned


def build_key_points(extracted_data: dict, content: str, session_id: str = "default") -> list[str]:
    """Use LLM to produce concise, non-paragraph key points about the file."""
    try:
        # Compact summary of extracted data for the prompt
        def truncate(val):
            s = str(val)
            return s if len(s) <= 500 else s[:497] + "..."

        lines = []
        for section, data in (extracted_data or {}).items():
            if section == "classification":
                continue
            if isinstance(data, dict):
                for k, v in data.items():
                    if v is not None and v != "" and v != []:
                        lines.append(f"{section}.{k}: {truncate(v)}")
        extracted_flat = "\n".join(lines[:60])
        content_truncated = content[:4000]

        prompt_template = PROMPTS["key_points_formatter"]
        prompt = prompt_template.format(
            extracted_flat=extracted_flat,
            content=content_truncated
        )
        
        with TokenTracker(
            operation="build_key_points",
            session_id=session_id,
            prompt=prompt,
            content=content_truncated
        ) as tracker:
            tracker.mark_generation_start()
            result = str(llm.invoke(prompt))
            tracker.mark_generation_done(result)
        
        points = [p.strip() for p in result.splitlines() if p.strip().startswith('- ')]
        # Fallback: if LLM returned a paragraph, split into sentences and prefix
        if not points:
            import re as _re
            sentences = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", result) if s.strip()]
            points = ["- " + s for s in sentences[:8]]
        return points[:10]
    except Exception:
        return ["- Key points unavailable"]


@app.post("/v1/chat", response_model=RunResponse)
def chat(message: str, session_id: str = "default"):
    """Single chat endpoint. Routes to general chat or file-aware analysis based on intent.
    Maintains rolling chat history (20 turns) per session in txt file.
    """
    if not logging.getLogger().handlers:
        log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.log")
        logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    total_start_time = time.time()
    
    history = load_history(session_id)
    # Build context: last turns + current user message
    context_messages = []
    for t in history:
        context_messages.append({"role": t.get("role", "user"), "content": t.get("content", "")})
    context_messages.append({"role": "user", "content": message})

    # Quick heuristic: if user references a path or code keywords, run analysis; else general chat via graph
    needs_analysis = any(k in message.lower() for k in [".py", ".ipynb", "class ", "def ", "import ", "model ", "accuracy", "optimizer", "dataset"]) or "file:" in message.lower() or "code:" in message.lower()

    if needs_analysis:
        with TokenTracker(
            operation="chat_analysis",
            session_id=session_id,
            prompt=message,
            content=message,
            chat_history=context_messages[:-1]  # Exclude current message from history count
        ) as tracker:
            processing_start = time.time()
            extracted = dynamic_extractor.extract_entities(message, "", session_id=session_id)
            validation = ai_validator.validate_extraction(extracted, message, session_id=session_id)
            key_points = build_key_points(extracted, message, session_id=session_id)
            grounded = suggestion_agent.build_grounded_suggestions(extracted, use_web=True, session_id=session_id)
            tracker.mark_processing_done()
            tracker.mark_generation_start()
            bot_reply = "\n".join(key_points + ["", "Suggestions:"] + grounded.get("suggestions", []))
            tracker.mark_generation_done(bot_reply)
    else:
        with TokenTracker(
            operation="chat_general",
            session_id=session_id,
            prompt="",
            content=message,
            chat_history=context_messages
        ) as tracker:
            processing_start = time.time()
            chat_info = ChatInfo()
            chat_info.session_id = session_id
            state = {"messages": context_messages, "chat_info": chat_info, "project_details": ProjectDetails()}
            tracker.mark_processing_done()
            tracker.mark_generation_start()
            out = graph.invoke(state)
            bot_reply = out["messages"][-1].content
            tracker.mark_generation_done(bot_reply)

    cleaned = clean_response(bot_reply)
    append_turn(session_id, "user", message)
    append_turn(session_id, "assistant", cleaned)
    
    total_time = time.time() - total_start_time
    
    # Log overall chat completion
    logger = logging.getLogger(__name__)
    logger.info(f"CHAT_COMPLETE | Session: {session_id} | Total time: {total_time:.3f}s | Analysis: {needs_analysis}")
    
    return RunResponse(reply=cleaned)


def _truth_check(validation: dict) -> bool:
    """Detect likely false info based on validation signals."""
    try:
        vr = validation.get("validation_results", {}) if isinstance(validation, dict) else {}
        hallucination_risk = (vr.get("hallucination_risk") or "").lower()
        conf = float(vr.get("confidence_score") or 0.0)
        acc = bool(vr.get("is_accurate") or False)
        comp = float(vr.get("completeness_score") or 0.0)
        return (hallucination_risk == "high") or (not acc and (conf < 0.5 or comp < 0.5))
    except Exception:
        return False


## Deprecated: standalone analyze in favor of single /v1/chat; keeping for compatibility optional


## Removed standalone suggestions endpoint in favor of unified /v1/analyze



if __name__ == "__main__":
    import uvicorn
    # Allow overriding via PORT env var; defaults to 8000
    _port = int(os.getenv("PORT", "8000"))
    # Run the FastAPI app with reload enabled for local development
    uvicorn.run("api.main:app", host="0.0.0.0", port=_port, reload=True)
