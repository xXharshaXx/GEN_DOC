## FastAPI service for LangGraph document generation

### Quickstart

1. Create a `.env` file next to this README:

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_GENERAL_MODEL=qwen3:4b-fp16
OLLAMA_CODE_MODEL=deepseek-coder-v2
LANGSMITH_API_KEY=
LANGFUSE_PUBLIC_KEY=pk-lf-84537175-e120-4da9-b417-771e1e1996f0
LANGFUSE_SECRET_KEY=sk-lf-6daac367-c5c5-437b-9569-100c91d769f0
LANGFUSE_HOST=https://cloud.langfuse.com
PORT=8000
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

**Note:** `tiktoken` is included in requirements.txt and does **NOT** require any API key. It's an open-source library that runs locally on your machine for token counting.

3. Run API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Or with auto-reload for development:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

4. Test the API:

```bash
# Run the test script
python test_api.py

# Or test manually with curl
curl -X POST "http://localhost:8000/v1/chat?message=Hello&session_id=test1"
```

### Endpoints

- **POST `/v1/chat`** → Chat endpoint with message analysis
  - Parameters:
    - `message` (str): The user message
    - `session_id` (str, optional): Session ID for chat history (default: "default")
  - Returns: `{"reply": "..."}`

### Features

- ✅ **Token Counting**: Uses `tiktoken` (open-source, no API key needed) to count input/output tokens
- ✅ **Time Tracking**: Logs processing time, generation time, and total time
- ✅ **Logging**: All token usage and timing information is logged to `app.log`
- ✅ **Chat History**: Maintains rolling chat history (20 turns) per session

### Logging

All operations log token usage and timing to `app.log`:

```
TOKEN_USAGE | extract_entities | Session: test_session | Input: 150 tokens | Output: 200 tokens | Total: 350 tokens | Processing: 0.120s | Generation: 1.450s | Total: 1.570s
```

### Viewing Logs

```bash
# View logs in real-time
tail -f app.log

# Or on Windows PowerShell
Get-Content app.log -Wait
```


