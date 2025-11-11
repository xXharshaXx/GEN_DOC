import json
import requests
from typing import List, Dict, Any
from .config import settings


def tavily_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    api_key = settings.tavily_api_key
    if not api_key:
        return []
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            data=json.dumps({"query": query, "max_results": max_results}),
            timeout=20,
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results") or data.get("data") or []
            return results[:max_results]
        return []
    except Exception:
        return []


