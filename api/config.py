import os
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()


class Settings(BaseModel):
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_general_model: str = os.getenv("OLLAMA_GENERAL_MODEL", "qwen3:4b-fp16")
    ollama_code_model: str = os.getenv("OLLAMA_CODE_MODEL", "deepseek-coder-v2")
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    langfuse_host: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "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")
    chat_history_dir: str = os.getenv("CHAT_HISTORY_DIR", "chat_history")


settings = Settings()


