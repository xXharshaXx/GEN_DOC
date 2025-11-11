# Setup and Testing Guide

## Tiktoken Configuration

✅ **Tiktoken is already configured and does NOT require any API key!**

- Tiktoken is an **open-source library** that runs locally on your machine
- It uses the `cl100k_base` encoding (same as GPT-4) for token counting
- No API keys, no external services needed
- Just install it with `pip install -r requirements.txt`

## Quick Setup

### 1. Install Dependencies

```bash
cd GEN_DOC_3.1
pip install -r requirements.txt
```

This will install:
- FastAPI and Uvicorn (web server)
- LangChain and LangGraph (AI framework)
- **Tiktoken** (token counting - no API key needed)
- Other dependencies

### 2. Set Up Environment Variables

Create a `.env` file in the `GEN_DOC_3.1` directory:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_GENERAL_MODEL=qwen3:4b-fp16
OLLAMA_CODE_MODEL=deepseek-coder-v2
LANGSMITH_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
PORT=8000
```

**Note:** Make sure Ollama is running on your machine if you're using it.

### 3. Run the Server

```bash
# From GEN_DOC_3.1 directory
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Or with auto-reload for development:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

## Testing the API

### Option 1: Use the Test Script

```bash
# Install requests if not already installed
pip install requests

# Run the test script
python test_api.py
```

### Option 2: Use curl (Command Line)

```bash
# Simple chat
curl -X POST "http://localhost:8000/v1/chat?message=Hello&session_id=test1"

# Code analysis (triggers token counting)
curl -X POST "http://localhost:8000/v1/chat?message=import%20torch%20def%20model():%20pass&session_id=test2"
```

### Option 3: Use Python Requests

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat",
    params={
        "message": "Analyze this code: def hello(): print('world')",
        "session_id": "my_session"
    }
)

print(response.json())
```

### Option 4: Use the Interactive API Docs

Open your browser and go to:
```
http://localhost:8000/docs
```

This will show an interactive Swagger UI where you can test the API endpoints directly.

## Checking Token Logs

After running requests, check the `app.log` file:

### Windows PowerShell:
```powershell
Get-Content app.log -Wait
```

### Windows CMD:
```cmd
type app.log
```

### Linux/Mac:
```bash
tail -f app.log
```

### Example Log Output:

```
2024-01-15 10:30:45 [INFO] token_utils: TOKEN_USAGE | classify_intent | Session: test_session | Input: 85 tokens | Output: 120 tokens | Total: 205 tokens | Processing: 0.000s | Generation: 0.450s | Total: 0.450s
2024-01-15 10:30:46 [INFO] token_utils: TOKEN_USAGE | extract_entities | Session: test_session | Input: 250 tokens | Output: 350 tokens | Total: 600 tokens | Processing: 0.000s | Generation: 1.200s | Total: 1.200s
2024-01-15 10:30:47 [INFO] token_utils: TOKEN_USAGE | validate_extraction | Session: test_session | Input: 400 tokens | Output: 150 tokens | Total: 550 tokens | Processing: 0.000s | Generation: 0.800s | Total: 0.800s
2024-01-15 10:30:48 [INFO] token_utils: TOKEN_USAGE | build_key_points | Session: test_session | Input: 300 tokens | Output: 200 tokens | Total: 500 tokens | Processing: 0.000s | Generation: 0.600s | Total: 0.600s
2024-01-15 10:30:49 [INFO] token_utils: TOKEN_USAGE | build_grounded_suggestions | Session: test_session | Input: 500 tokens | Output: 250 tokens | Total: 750 tokens | Processing: 0.000s | Generation: 1.100s | Total: 1.100s
2024-01-15 10:30:50 [INFO] main: CHAT_COMPLETE | Session: test_session | Total time: 4.250s | Analysis: True
```

## What Gets Logged?

For each operation, the system logs:

1. **Input Tokens**: 
   - Prompt tokens
   - Content tokens
   - Reference content tokens (from web search if used)
   - Chat history tokens (if used)

2. **Output Tokens**: 
   - Generated response tokens

3. **Timing**:
   - **Processing Time**: Time spent preparing/processing data before LLM call
   - **Generation Time**: Time spent in LLM generation
   - **Total Time**: Complete operation time

4. **Metadata**:
   - Operation name (e.g., `extract_entities`, `validate_extraction`)
   - Session ID
   - File path (if applicable)

## Troubleshooting

### Tiktoken Import Error
If you get an error about tiktoken:
```bash
pip install tiktoken>=0.5.0
```

### Server Won't Start
- Check if port 8000 is already in use
- Make sure all dependencies are installed
- Check `.env` file exists and has correct values

### No Logs Appearing
- Make sure the `GEN_DOC_3.1` directory is writable
- Check if `app.log` file exists (it will be created automatically)
- Verify logging is enabled (it should be by default)

### Token Counts Seem Wrong
- Tiktoken uses `cl100k_base` encoding (same as GPT-4)
- If you're using a different model, token counts may vary slightly
- The fallback method (characters/4) is used if tiktoken fails

## Next Steps

- Test different types of messages (code, general chat, file analysis)
- Check the logs to see token usage patterns
- Monitor processing times for optimization
- Use different session_ids to maintain separate chat histories

