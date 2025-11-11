"""
Utility functions for token counting and time tracking
"""
import tiktoken
import time
import logging
import os
from typing import Dict, Any, List, Optional


# Initialize tiktoken encoder (using cl100k_base which works for most models)
try:
    encoding = tiktoken.get_encoding("cl100k_base")
except Exception:
    encoding = None


def count_tokens(text: str) -> int:
    """Count tokens in a text string using tiktoken"""
    if not text or not encoding:
        # Fallback: approximate 4 characters per token
        return len(str(text)) // 4 if text else 0
    try:
        return len(encoding.encode(str(text)))
    except Exception:
        # Fallback: approximate 4 characters per token
        return len(str(text)) // 4


def count_tokens_in_messages(messages: List[Dict[str, Any]]) -> int:
    """Count tokens in a list of messages (chat history)"""
    total = 0
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            total += count_tokens(f"{role}: {content}")
        else:
            # Handle langchain message objects
            try:
                content = getattr(msg, "content", str(msg))
                total += count_tokens(str(content))
            except Exception:
                total += count_tokens(str(msg))
    return total


def count_tokens_in_prompt(prompt: str, content: str = "", reference_content: str = "", chat_history: List[Dict[str, Any]] = None) -> int:
    """Count total input tokens including prompt, content, reference content, and chat history"""
    total = count_tokens(prompt)
    total += count_tokens(content)
    total += count_tokens(reference_content)
    if chat_history:
        total += count_tokens_in_messages(chat_history)
    return total


def log_token_usage(
    operation: str,
    input_tokens: int,
    output_tokens: int,
    processing_time: float,
    generation_time: float,
    total_time: float,
    session_id: str = "default",
    file_path: str = "",
    additional_info: Dict[str, Any] = None
):
    """Log token usage and timing information to app.log"""
    if not logging.getLogger().handlers:
        log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.log")
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
    
    logger = logging.getLogger(__name__)
    
    log_data = {
        "operation": operation,
        "session_id": session_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "processing_time_sec": round(processing_time, 3),
        "generation_time_sec": round(generation_time, 3),
        "total_time_sec": round(total_time, 3),
    }
    
    if file_path:
        log_data["file_path"] = file_path
    
    if additional_info:
        log_data.update(additional_info)
    
    log_message = f"TOKEN_USAGE | {operation} | Session: {session_id}"
    if file_path:
        log_message += f" | File: {file_path}"
    log_message += f" | Input: {input_tokens} tokens | Output: {output_tokens} tokens | Total: {input_tokens + output_tokens} tokens"
    log_message += f" | Processing: {processing_time:.3f}s | Generation: {generation_time:.3f}s | Total: {total_time:.3f}s"
    
    logger.info(log_message)


class TokenTracker:
    """Context manager for tracking tokens and time"""
    
    def __init__(self, operation: str, session_id: str = "default", file_path: str = "", prompt: str = "", content: str = "", reference_content: str = "", chat_history: List[Dict[str, Any]] = None):
        self.operation = operation
        self.session_id = session_id
        self.file_path = file_path
        self.prompt = prompt
        self.content = content
        self.reference_content = reference_content
        self.chat_history = chat_history or []
        self.start_time = None
        self.processing_start = None
        self.generation_start = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.processing_time = 0.0
        self.generation_time = 0.0
        self.total_time = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        self.processing_start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.total_time = time.time() - self.start_time
        
        # Count input tokens
        self.input_tokens = count_tokens_in_prompt(
            self.prompt,
            self.content,
            self.reference_content,
            self.chat_history
        )
        
        # Log the usage
        log_token_usage(
            operation=self.operation,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            processing_time=self.processing_time,
            generation_time=self.generation_time,
            total_time=self.total_time,
            session_id=self.session_id,
            file_path=self.file_path
        )
    
    def mark_processing_done(self):
        """Mark the end of processing phase"""
        if self.processing_start:
            self.processing_time = time.time() - self.processing_start
    
    def mark_generation_start(self):
        """Mark the start of generation phase"""
        self.mark_processing_done()
        self.generation_start = time.time()
    
    def mark_generation_done(self, output_text: str = ""):
        """Mark the end of generation phase and count output tokens"""
        if self.generation_start:
            self.generation_time = time.time() - self.generation_start
        if output_text:
            self.output_tokens = count_tokens(output_text)

